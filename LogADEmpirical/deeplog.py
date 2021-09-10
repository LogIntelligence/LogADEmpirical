import os
import pickle

from LogADEmpirical.logdeep.models.lstm import deeplog
from LogADEmpirical.logdeep.tools.predict import Predicter
from LogADEmpirical.logdeep.tools.train import Trainer
from LogADEmpirical.logdeep.dataset.vocab import Vocab


def run_deeplog(options):
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
        vocab = Vocab(logs, os.path.join(options['data_dir'], "embeddings.json"), "deeplog")
        print("vocab size", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])
    Trainer(options).start_train()
    Predicter(options).predict_semi_supervised()
