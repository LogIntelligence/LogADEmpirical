import os
from logbert.logdeep.models.lstm import loganomaly
from logbert.logdeep.tools.predict import Predicter
from logbert.logdeep.tools.train import Trainer
from logbert.logdeep.dataset.vocab import Vocab
import pickle


def run_loganomaly(options):
    if not os.path.exists(options["vocab_path"]):
        with open(options["train_vocab"], 'rb') as f:
            data = pickle.load(f)
        logs = []
        for x in data:
            try:
                l = max(x['Label'])
            except:
                l = x['Label']
            if l == 0:
                logs.append(x['EventId'])
        vocab = Vocab(logs)
        print("vocab size", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])

    Trainer(options).start_train()
    Predicter(options).predict_semi_supervised()
