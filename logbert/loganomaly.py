import os
from logbert.logdeep.models.lstm import loganomaly
from logbert.logdeep.tools.predict import Predicter
from logbert.logdeep.tools.train import Trainer
from logbert.logdeep.dataset.vocab import Vocab


def run_loganomaly(options):
    if not os.path.exists(options["vocab_path"]):
        with open(options["train_vocab"], 'r') as f:
            logs = f.readlines()
        vocab = Vocab(logs)
        print("vocab size", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])

    Trainer(options).start_train()
    Predicter(options).predict_semi_supervised()
