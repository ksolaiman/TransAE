import numpy as np
import torch
import torch.nn as nn

import numpy as np
import pickle
import gensim

            
def pvdm(entity2id, retrain=True, vector_size=100, min_count=2, epochs=40):
    if not retrain:
        with open("../embedding_weights/textembed_" + str(vector_size) + "_" + str(min_count) + "_" + str(epochs) +".pkl", "rb") as emf:
            inferred_vector_list = pickle.load(emf)
        return inferred_vector_list
    
    entity2glossary = dict()
    with open("../WN9-IMG_IKRL/gloss.txt", "r") as glossf: # TODO: add datapath
        for line in glossf:
            # print(line)
            entity, glossary = line.split("\t")
            entity2glossary[entity] = glossary
    
    entity2description = list()
    # Was Doing training on whole dataset, should not do it, should be done only on training dataset
    for entity, index in entity2id.items():
        entity2description.append(entity2glossary[entity])

    def read_corpus(tokens_only=False):
        for i, v in enumerate(entity2description):
            tokens = gensim.utils.simple_preprocess(v)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
            
    train_corpus = list(read_corpus())
    
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    inferred_vector_list = list()

    for doc_id in range(len(train_corpus)): # train_corpus is already sorted in entity2id order, so will be the saved vectors
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        # print(inferred_vector)     # inferred_vector is of size embedding_dim
        inferred_vector_list.append(inferred_vector)
        
    with open("../embedding_weights/textembed_" + str(vector_size) + "_" + str(min_count) + "_" + str(epochs) +".pkl", "wb+") as emf:
        pickle.dump(inferred_vector_list, emf)
    
    return inferred_vector_list

class AutoEncoder(torch.nn.Module):
    # TODO: nn.Embedding(num_embeddings=self.entity_count + 1 !!, embedding_dim=self.dim, padding_idx=self.entity_count !!)
    def __init__(self, entity2id, No_of_entities, text_embedding_dim=100, visual_embedding_dim=4096, 
                 hidden_text_dim=50, hidden_visual_dim=512, hidden_dimension=50, activation='sigmoid'):
        """
        In the constructor we instantiate the modules and assign them as
        member variables.
        """
        super(AutoEncoder, self).__init__()
        
        self.entity2id = entity2id
        self.activation = nn.Sigmoid()
        self.entity_count = len(entity2id)
        self.dim = hidden_text_dim
        self.text_embedding_dim = text_embedding_dim
        self.visual_embedding_dim = visual_embedding_dim
        self.criterion = nn.MSELoss(reduction='mean')        # L2 loss
        
        # output from following two layers are v_1's
        # input layer
        ## self.text_embedding = nn.Embedding(No_of_entities, text_embedding_dim)
        ## self.visual_embedding = nn.Embedding(No_of_entities, visual_embedding_dim)
        self.text_embedding = self._init_text_emb()
        self.visual_embedding = self._init_visual_emb()
        
        # hidden layer 1
        self.encoder_text_linear1 = nn.Sequential(
            torch.nn.Linear(text_embedding_dim, hidden_text_dim),
            self.activation
        )
        self.encoder_visual_linear1 = nn.Sequential(
            torch.nn.Linear(visual_embedding_dim, hidden_visual_dim),
            self.activation
        )
        
        # hidden layer 2
        # hidden_dimension is same dimension as the relation embedding dim, which is just dim in TransE model
        self.encoder_combined_linear = nn.Sequential(
            torch.nn.Linear(hidden_text_dim + hidden_visual_dim, hidden_dimension),
            self.activation
        )
        
        # hidden layer 3
        # each shares the same dimension/ have the same output dimension with the corresponding hidden 
        # layer (hidden layer 1) in the encoder part
        self.decoder_text_linear1 = nn.Sequential(
            torch.nn.Linear(hidden_dimension, hidden_text_dim),
            self.activation
        )
        self.decoder_visual_linear1 = nn.Sequential(
            torch.nn.Linear(hidden_dimension, hidden_visual_dim),
            self.activation
        )
        
        # output layer
        # the output layer and input layer are of the same dimension for each modality
        # output from following two layers are v_5's
        self.decoder_text_linear2 = nn.Sequential(
            torch.nn.Linear(hidden_text_dim, text_embedding_dim),
            self.activation
        )
        self.decoder_visual_linear2 = nn.Sequential(
            torch.nn.Linear(hidden_visual_dim, visual_embedding_dim),
            self.activation
        )
        
        
    def _init_text_emb(self):
        inferred_vector_list = pvdm(self.entity2id, retrain=False) # should just train on the ones in training set
        weights = np.zeros((self.entity_count + 1, self.text_embedding_dim)) # +1 to account for padding/OOKB, initialized to 0 each one
        for index in range(len(inferred_vector_list)):
            weights[index] = inferred_vector_list[index]
        weights = torch.from_numpy(weights) 
        # print(weights.shape)
        text_emb = nn.Embedding.from_pretrained(weights, padding_idx=self.entity_count)
        return text_emb
    
    # TODO: initialize visual embedding layers
    def _init_visual_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.visual_embedding_dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)         # Equn 16 in the Xavier initialization paper from TransE paper
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb

        
    # can pass entity_id or entity_name(in that case has to do a lookup in forward)
    def forward(self, entity_id_tensors):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # For batch training a group of entity_id will be passed
        v1_t = self.text_embedding(entity_id_tensors).float()
        v1_i = self.visual_embedding(entity_id_tensors)
        
        v2_t = self.encoder_text_linear1(v1_t)
        v2_i = self.encoder_visual_linear1(v1_i)
        
        v3 = self.encoder_combined_linear(torch.cat((v2_t, v2_i), 1))
        ### print(v3.size())
        ### print(torch.cat((v2_t, v2_i), 1).size())
        
        v4_t = self.decoder_text_linear1(v3)
        v4_i = self.decoder_visual_linear1(v3)
        
        v5_t = self.decoder_text_linear2(v4_t)
        v5_i = self.decoder_visual_linear2(v4_i)
        
        ### print(v5_t.size())
        ### print(v5_i.size())
        
        
        # should happen for each entity call, either positive or negative sample, does not matter, so has to be done here
        # can use the following only when loss has mean/sum as 'reduction' defined
        recon_error = self.criterion(v1_t, v5_t) + self.criterion(v1_i, v5_i)
        ## or, the following
        ## a = torch.cat((v1_t, v1_i), dim=1) 
        ## b = torch.cat((v5_t, v5_i), dim=1) 
        ## recon_error = self.criterion(a, b)
        print(recon_error)
        
        return v3, recon_error
        
        
class TransE(nn.Module):

    def __init__(self, entity2id, entity_count, relation_count, device, norm=1, dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity2id = entity2id           # for autoencoder embedding layer weight initialization
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim                                # probably d in the TransAE paper
        self.entities_emb = self._init_enitity_emb()
        self.autoencoder = AutoEncoder(self.entity2id, self.entity_count)
        # self.entities_emb = self.autoencoder.encoder_combined_linear
        self.relations_emb = self._init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='mean') # replaced reduction='none', as it makes it easier to add reconstruction loss

    def _init_enitity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)         # Equn 16 in the Xavier initialization paper from TransE paper
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))  # normalizing with abs value of weight data
        return relations_emb

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        """Return model losses based on the input.

        :param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        #TODO: This line needs to be fixed
        # -1 to avoid nan for OOV vector
        self.entities_emb.weight.data[:-1, :].div_(self.entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        assert positive_triplets.size()[1] == 3
        positive_distances, recon_loss_pos = self._distance(positive_triplets)

        assert negative_triplets.size()[1] == 3
        negative_distances, recon_loss_neg = self._distance(negative_triplets)

        # DONE: may have to change TransE loss function, add reduction as mean, instead of doint it in main func
        # then add to self.loss(..) + recon_loss_pos + recon_loss_neg
        return self.loss(positive_distances, negative_distances) + recon_loss_pos + recon_loss_neg, positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        v3_h, recon_loss_h = self.autoencoder(heads) 
        v3_t, recon_loss_t = self.autoencoder(tails)
        # input("wait")
        # return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm, dim=1)
        return (v3_h + self.relations_emb(relations) - v3_t).norm(p=self.norm, dim=1), recon_loss_h + recon_loss_t
