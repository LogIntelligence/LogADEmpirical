import os
import pickle

from logadempirical.logdeep.tools.predict import Predicter
from logadempirical.logdeep.tools.train import Trainer
from logadempirical.logdeep.dataset.vocab import Vocab


def run_logrobust(options):
    if not os.path.exists(options["vocab_path"]):
        with open(options["train_vocab"], 'rb') as f:
            data = pickle.load(f)
        logs = [x['EventId'] for x in data]
        vocab = Vocab(logs, os.path.join(options['data_dir'], options["embeddings"]), "logrobust")
        print("vocab size", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])

    Trainer(options).start_train()
    Predicter(options).predict_supervised2()
