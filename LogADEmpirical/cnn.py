import os

from LogADEmpirical.logdeep.models.lstm import  deeplog
from LogADEmpirical.logdeep.tools.predict import Predicter
from LogADEmpirical.logdeep.tools.train import Trainer
from LogADEmpirical.logdeep.dataset.vocab import Vocab
import pickle



def run_cnn(options):
    if not os.path.exists(options["vocab_path"]):
        with open(options["train_vocab"], 'rb') as f:
            data = pickle.load(f)
        logs = [x['EventId'] for x in data]
        vocab = Vocab(logs, os.path.join(options['data_dir'], options["embeddings"]), "cnn")
        print("vocab size", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])

    Trainer(options).start_train()
    Predicter(options).predict_supervised2()
