from collections import Counter
from torch.utils import data
from typing import Dict, Tuple
import os

Mapping = Dict[str, int]

def create_mappings_for_W9(path: str) -> Tuple[Mapping, Mapping]:
    """Creates separate mappings to indices for entities and relations."""
    entity2id_path = os.path.join(path, "entity2id.txt")
    relation2id_path = os.path.join(path, "relation2id.txt")
    entity2id = {}
    relation2id = {}
    with open(entity2id_path, "r") as f:
        for line in f:
            # -1 to remove newline sign
            entity, idx  = line[:-1].split("\t")
            entity2id[entity] = idx
            
    with open(relation2id_path, "r") as f:
        for line in f:
            # -1 to remove newline sign
            relation, idx  = line[:-1].split("\t")
            relation2id[entity] = idx
    
    return entity2id, relation2id

def create_mappings(dataset_path: str) -> Tuple[Mapping, Mapping]:
    """Creates separate mappings to indices for entities and relations."""
    # counters to have entities/relations sorted from most frequent
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r") as f:
        for line in f:
            # -1 to remove newline sign
            head, relation, tail  = line[:-1].split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    return entity2id, relation2id


class FB15KDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            # data in tuples (head, relation, tail)
            self.data = [line[:-1].split("\t") for line in f]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)
        
# https://pytorch.org/docs/stable/data.html?highlight=dataloader
# 2 types of data, map and iter., following is map
class W9Dataset(data.Dataset):
    """Dataset implementation for handling W9-IMG-TXT."""

    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        with open(data_path, "r") as f:
            # data in tuples (head, relation, tail)
            self.data = [line[:-1].split("\t") for line in f]

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    # The __getitem__ method uses an index to get a single samples not a batch, i.e. the batch dimension of your data is missing in __getitem__.
    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        # changed to head, tail, relation from head, relation, tail for W9 dataset
        head, tail, relation = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)
