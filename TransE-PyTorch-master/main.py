from absl import app
from absl import flags
import data
import metric
import model as model_definition
import os
import storage
import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard
from typing import Tuple

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.01, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=128, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64, help="Maximum batch size during model validation.")
flags.DEFINE_integer("vector_length", default=50, help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0, help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer("norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=2000, help="Number of training epochs.")
flags.DEFINE_string("dataset_path", default="./synth_data", help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")
flags.DEFINE_integer("validation_freq", default=10, help="Validate model every X epochs.")
flags.DEFINE_string("checkpoint_path", default="", help="Path to model checkpoint (by default train from scratch).")
flags.DEFINE_string("tensorboard_log_dir", default="./runs", help="Path for tensorboard log directory.")

flags.DEFINE_string("en_reln_mapping", default="file", help="Function to use for creating embedding mappings.") # 'file', 'wn9', 'fb15k'
flags.DEFINE_bool("retrain_text_layer", default=False, help="Retrain the PV-DM model.") 
flags.DEFINE_bool("test_only", default=False, help="Just running the saved model on test dataset.") 
flags.DEFINE_float("beta", default=0.4, help="weight of margin ranking loss.")

HITS_AT_1_SCORE = float
HITS_AT_3_SCORE = float
HITS_AT_10_SCORE = float
MRR_SCORE = float
METRICS = Tuple[HITS_AT_1_SCORE, HITS_AT_3_SCORE, HITS_AT_10_SCORE, MRR_SCORE]


def test(model: torch.nn.Module, data_generator: torch_data.DataLoader, entities_count: int,
         summary_writer: tensorboard.SummaryWriter, device: torch.device, epoch_id: int, metric_suffix: str,
         ) -> METRICS:
    examples_count = 0.0
    hits_at_1 = 0.0
    hits_at_3 = 0.0
    hits_at_10 = 0.0
    mrr = 0.0

    entity_ids = torch.arange(end=entities_count, device=device).unsqueeze(0) # Returns a 1-D tensor of size entities_count with values from 0 to entities_count, and then Returns a new tensor with a dimension of size one inserted at the specified position/ basically adding another dimension.
    for head, relation, tail in data_generator:
        # print(head, relation, tail)
        current_batch_size = head.size()[0]

        head, relation, tail = head.to(device), relation.to(device), tail.to(device)
        all_entities = entity_ids.repeat(current_batch_size, 1) # with torch.repeat(), you can specify the number of repeats for each dimension
        heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails = tail.reshape(-1, 1).repeat(1, all_entities.size()[1])
        
        # Check all possible tails
        triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        # print(triplets)
        tails_predictions = model.predict(triplets).reshape(current_batch_size, -1)
        # Check all possible heads
        triplets = torch.stack((all_entities, relations, tails), dim=2).reshape(-1, 3)
        heads_predictions = model.predict(triplets).reshape(current_batch_size, -1)

        # Concat predictions
        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((tail.reshape(-1, 1), head.reshape(-1, 1)))
        
        # Each prediction is an array of N size, where N is no_of_Entity_in_KB, and there are no_of_samples_in_batch * 2 (head & tail) ground_truth and column level prediction

        # https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
        hits_at_1 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=1)
        hits_at_3 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=3)
        hits_at_10 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=10)
        mrr += metric.mrr(predictions, ground_truth_entity_id)

        examples_count += predictions.size()[0]

    hits_at_1_score = hits_at_1 / examples_count * 100
    hits_at_3_score = hits_at_3 / examples_count * 100
    hits_at_10_score = hits_at_10 / examples_count * 100
    mrr_score = mrr / examples_count * 100
    summary_writer.add_scalar('Metrics/Hits_1/' + metric_suffix, hits_at_1_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/Hits_3/' + metric_suffix, hits_at_3_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/Hits_10/' + metric_suffix, hits_at_10_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/MRR/' + metric_suffix, mrr_score, global_step=epoch_id)

    return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score


def collate_fn(data):
    # print(data) # prints one batch of data, i.e., [('4684', 1, '1363'), ('2814', 1, '5384'), ('3045', 1, '3039')]
    heads, relations, tails = list(), list(), list()
    for i in range(len(data)):
        heads.append(int(data[i][0]))
        relations.append(int(data[i][1]))
        tails.append(int(data[i][2])) # coz they are string
    return torch.tensor(heads), torch.tensor(relations), torch.tensor(tails)

def main(_):
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path = FLAGS.dataset_path
    train_path = os.path.join(path, "train.txt")
    validation_path = os.path.join(path, "valid.txt")
    test_path = os.path.join(path, "test.txt")

    if FLAGS.en_reln_mapping is 'file':
        entity2id, relation2id = data.create_mappings_for_WN9(path)
    elif FLAGS.en_reln_mapping is 'wn9':
        entity2id, relation2id = data.create_mappings(train_path, 'WN9')
    elif FLAGS.en_reln_mapping is 'fb15k':
        entity2id, relation2id = data.create_mappings(train_path, 'FB15K')
    else:
        entity2id, relation2id = data.create_mappings(train_path, 'WN9')
    
    
    #for key, value in sorted(entity2id.items(), key=lambda x: x[1]): 
    #    print("{} : {}".format(key, value))   # No OOKB entityid for now
        # input("wait")

    batch_size = FLAGS.batch_size
    vector_length = FLAGS.vector_length
    margin = FLAGS.margin
    norm = FLAGS.norm
    learning_rate = FLAGS.lr
    epochs = FLAGS.epochs
    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')

    # train_set = data.FB15KDataset(train_path, entity2id, relation2id)
    train_set = data.WN9Dataset(train_path, entity2id, relation2id)
    # collate_fn (callable, optional) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn)
    # validation_set = data.FB15KDataset(validation_path, entity2id, relation2id)
    validation_set = data.WN9Dataset(validation_path, entity2id, relation2id)
    validation_generator = torch_data.DataLoader(validation_set, batch_size=FLAGS.validation_batch_size, collate_fn=collate_fn)
    # test_set = data.FB15KDataset(test_path, entity2id, relation2id)
    test_set = data.WN9Dataset(test_path, entity2id, relation2id)
    test_generator = torch_data.DataLoader(test_set, batch_size=FLAGS.validation_batch_size, collate_fn=collate_fn)

    autoencoder =  model_definition.AutoEncoder(entity2id, retrain_text_layer=FLAGS.retrain_text_layer, hidden_dimension=vector_length) # for autoencoder embedding layer weight initialization
    model = model_definition.TransE(entity_count=len(entity2id), relation_count=len(relation2id), dim=vector_length,
                                    margin=margin, beta=FLAGS.beta,
                                    device=device, norm=norm,
                                    autoencoder=autoencoder)  # type: torch.nn.Module
    model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    summary_writer = tensorboard.SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)
    start_epoch_id = 1
    step = 0
    best_score = 0.0

    if FLAGS.checkpoint_path:
        start_epoch_id, step, best_score = storage.load_checkpoint(FLAGS.checkpoint_path, model, optimizer)

    print(model)

    if FLAGS.test_only:
        epochs = -1
        start_epoch_id = 0
    # Training loop
    for epoch_id in range(start_epoch_id, epochs + 1):
        print("Starting epoch: ", epoch_id)
        loss_impacting_samples_count = 0
        samples_count = 0
        model.train()

        for local_heads, local_relations, local_tails in train_generator:
            # data was not well prepared, had to use collate_fn in Dataloaders to fix it, will find it above
            local_heads, local_relations, local_tails = (local_heads.to(device), local_relations.to(device),
                                                         local_tails.to(device))

            positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1)

            # Preparing negatives.
            # Generate binary tensor to replace either head or tail. 1 means replace head, 0 means replace tail.
            head_or_tail = torch.randint(high=2, size=local_heads.size(), device=device)
            random_entities = torch.randint(high=len(entity2id), size=local_heads.size(), device=device)
            broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
            broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
            negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1)

            optimizer.zero_grad()

            loss, pd, nd = model(positive_triples, negative_triples)
            # loss.mean().backward()
            loss.backward()
    
            summary_writer.add_scalar('Loss/train', loss.data.cpu().numpy(), global_step=step)
#           summary_writer.add_scalar('Loss/train', loss.mean().data.cpu().numpy(), global_step=step)
            summary_writer.add_scalar('Distance/positive', pd.sum().data.cpu().numpy(), global_step=step)
            summary_writer.add_scalar('Distance/negative', nd.sum().data.cpu().numpy(), global_step=step)

            loss = loss.data.cpu()
            #loss_impacting_samples_count += loss.nonzero().size()[0]
            #samples_count += loss.size()[0]

            optimizer.step()
            step += 1

        #summary_writer.add_scalar('Metrics/loss_impacting_samples', loss_impacting_samples_count / samples_count * 100,
        #                          global_step=epoch_id)

        if epoch_id % FLAGS.validation_freq == 0:
            model.eval()
            _, _, hits_at_10, _ = test(model=model, data_generator=validation_generator,
                                       entities_count=len(entity2id),
                                       device=device, summary_writer=summary_writer,
                                       epoch_id=epoch_id, metric_suffix="val")
            score = hits_at_10
            print(score)
            if score > best_score:
                best_score = score
                storage.save_checkpoint(model, optimizer, epoch_id, step, best_score)

    # Testing the best checkpoint on test dataset
    storage.load_checkpoint("checkpoint.tar", model, optimizer)
    best_model = model.to(device)
    best_model.eval()
    scores = test(model=best_model, data_generator=test_generator, entities_count=len(entity2id), device=device,
                  summary_writer=summary_writer, epoch_id=1, metric_suffix="test")
    print("Test scores: ", scores)


if __name__ == '__main__':
    app.run(main)
