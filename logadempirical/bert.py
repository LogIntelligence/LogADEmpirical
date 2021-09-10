import os
from logadempirical.bert_pytorch.dataset import WordVocab
from logadempirical.bert_pytorch import Predictor, Trainer


def run_logbert(options):
    if not os.path.exists(options["vocab_path"]):
        with open(options["train_vocab"], "r") as f:
            texts = f.readlines()
        vocab = WordVocab(texts, min_freq=options["min_freq"])
        print("VOCAB SIZE:", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])

    Trainer(options).train()
    Predictor(options).predict()







