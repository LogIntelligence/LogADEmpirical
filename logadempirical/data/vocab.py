from collections import Counter
import pickle
import json
from numpy import dot
from numpy.linalg import norm


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


class Vocab(object):
    def __init__(self, logs, emb_file="embeddings.json"):
        self.pad_index = 0
        self.pad_token = "padding"

        self.stoi = {}
        self.itos = list(["padding"])

        event_count = Counter()
        for line in logs:
            for logkey in line:
                event_count[logkey] += 1

        for event in event_count.keys():
            self.itos.append(event)
        self.unk_index = len(self.itos)
        self.stoi = {e: i for i, e in enumerate(self.itos)}
        self.semantic_vectors = read_json(emb_file)
        self.semantic_vectors[self.pad_token] = [-1] * len(self.semantic_vectors[self.itos[1]])
        self.mapping = {}

    def __len__(self):
        return len(self.itos)

    def get_event(self, real_event, use_similar=False):
        event = self.stoi.get(real_event, self.unk_index)
        if not use_similar or event != self.unk_index:
            return event
        if real_event in self.mapping:
            return self.mapping[real_event]

        for train_event in self.itos[1:]:
            sim = dot(self.semantic_vectors[real_event], self.semantic_vectors[train_event]) / (norm(
                self.semantic_vectors[real_event]) * norm(self.semantic_vectors[train_event]))
            if sim > 0.90:
                self.mapping[real_event] = self.stoi.get(train_event)
                return self.stoi.get(train_event)
        self.mapping[real_event] = self.unk_index
        return self.mapping[real_event]

    def get_embedding(self, event):
        return self.semantic_vectors[event]

    def save_vocab(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_vocab(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
