import os

from logbert.logdeep.models.lstm import  deeplog
from logbert.logdeep.tools.predict import Predicter
from logbert.logdeep.tools.train import Trainer
from logbert.logdeep.dataset.vocab import Vocab



def run_cnn(options):
    if not os.path.exists(options["vocab_path"]):
        with open(options["train_vocab"], 'r') as f:
            logs = f.readlines()
        vocab = Vocab(logs)
        print("vocab size", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])

    # model = deeplog(input_size=options['input_size'],
    #                 hidden_size=options['hidden_size'],
    #                 num_layers=options['num_layers'],
    #                 vocab_size=options["vocab_size"],
    #                 embedding_dim=options["embedding_dim"])

    Trainer(options).start_train()
    Predicter(options).predict_supervised()