from collections import Counter
import pickle
import json
import os
from numpy import dot
from numpy.linalg import norm


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

class Vocab(object):
    def __init__(self, logs, emb_file="embeddings.json", model="deeplog"):
        self.pad_index = 0

        self.stoi = {}
        self.itos = list(["padding"])

        event_count = Counter()
        for line in logs:
            for logkey in line:
                event_count[logkey] += 1

        for event, freq in event_count.items():
            self.itos.append(event)
        self.unk_index = len(self.itos)
        self.stoi = {e: i for i, e in enumerate(self.itos)}
        # self.semantic_vectors = read_json(os.path.join(emb_file))
        # self.semantic_vectors["padding"] = [-1] * 300
        self.model = model
        self.mapping = {}

    def __len__(self):
        return len(self.itos)

    def find_similar(self, real_event):
        if real_event in self.mapping:
            return self.mapping[real_event]

        if self.model == "loganomaly":
            for train_event in self.itos:
                sim = dot(self.semantic_vectors[real_event], self.semantic_vectors[train_event])/(norm(
                    self.semantic_vectors[real_event])*norm(self.semantic_vectors[train_event]))
                if sim > 0.90:
                    self.mapping[real_event] = self.stoi.get(train_event)
                    return self.stoi.get(train_event)
        self.mapping[real_event] = self.unk_index
        return self.unk_index

    def save_vocab(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_vocab(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

