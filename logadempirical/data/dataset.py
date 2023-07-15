# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
from logadempirical.data.vocab import Vocab


class LogDataset(Dataset):
    def __init__(self, sequentials=None, quantitatives=None, semantics=None, labels=None, idxs=None):
        if sequentials is None and quantitatives is None and semantics is None:
            raise ValueError('Provide at least one feature type')
        self.sequentials = sequentials
        self.quantitatives = quantitatives
        self.semantics = semantics
        self.labels = labels
        self.idxs = idxs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {'label': self.labels[idx], 'idx': self.idxs[idx]}
        if self.sequentials is not None:
            item['sequential'] = torch.from_numpy(np.array(self.sequentials[idx]))
        if self.quantitatives is not None:
            item['quantitative'] = torch.from_numpy(np.array(self.quantitatives[idx], )[:, np.newaxis]).float()
        if self.semantics is not None:
            item['semantic'] = torch.from_numpy(np.array(self.semantics[idx])).float()

        return item
class LogDatase_Bert(Dataset):
    def __init__(self, log_corpus,vocab , seq_len, corpus_lines=None, encoding="utf-8", on_memory=True, predict_mode=False, mask_ratio=0.75 , idx=None):
        """

        :param corpus: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param corpus_lines: number of log sessions
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        self.vocab = vocab
        self.seq_len = seq_len
        self.idxs = idx
        self.on_memory = on_memory
        self.encoding = encoding
        self.predict_mode = predict_mode
        self.log_corpus = log_corpus
        self.corpus_lines = len(log_corpus)
        self.mask_ratio = mask_ratio
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        k= self.log_corpus[idx]

        k_masked, k_label = self.random_item(k)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        k = [self.sos_index] + k_masked
        k_label = [self.pad_index] + k_label
        # item = {"label":k_label ,'idx': self.idxs[idx] , "sequential":k}

        return k , idx, k_label 
    def random_item(self, k,):
        tokens = list(k)
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            # replace 65% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                # raise AttributeError("no mask in visualization")

                if self.predict_mode:
                    tokens[i] = self.mask_index
                    output_label.append(token)
                    continue
                prob /= self.mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = token

                output_label.append(token)
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        lens = [len(seq[0]) for seq in batch]
        
        # find the max len in each batch
        if dynamical_pad:
            # dynamical padding
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            # fixed length padding
            seq_len = self.seq_len

        output = {"sequential":[] , "label" :[] , "idx":[]}
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_idx = seq[1]
            bert_label = seq[2][:seq_len]


            padding = [self.vocab.pad_index for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding)
            output["sequential"].append(bert_input)
            output["label"].append(bert_label)
            output["idx"].append(bert_idx)

        output["sequential"] = torch.tensor(np.array(output["sequential"]), dtype=torch.long)
        output["label"] = torch.tensor(np.array(output["label"]), dtype=torch.long)
        output["idx"] = torch.tensor(np.array(output["idx"]), dtype=torch.long)


        return output
        


def data_collate(batch, feature_name='semantic', padding_side="right"):
    max_length = max([len(b[feature_name]) for b in batch])
    dimension = {k: batch[0][k][0].shape[0] for k in batch[0].keys() if k != 'label' and batch[0][k] is not None}
    if padding_side == "right":
        padded_batch = []
        for b in batch:
            sample = {}
            for k, v in b.items():
                if k == 'label':
                    sample[k] = v
                elif v is None:
                    sample[k] = None
                else:
                    sample[k] = torch.from_numpy(
                        np.array(v + [np.zeros(dimension[k], )] * (max_length - len(v))))
            padded_batch.append(sample)
    elif padding_side == "left":
        padded_batch = []
        for b in batch:
            sample = {}
            for k, v in b.items():
                if k == 'label':
                    sample[k] = v
                elif v is None:
                    sample[k] = None
                else:
                    sample[k] = torch.from_numpy(
                        np.array([np.zeros(dimension[k], )] * (max_length - len(v)) + v))
            padded_batch.append(sample)
    else:
        raise ValueError("padding_side should be either 'right' or 'left'")

    # convert to tensor
    padded_batch = {
        k: torch.stack([sample[k] for sample in padded_batch])
        for k in padded_batch[0].keys()
    }
    return padded_batch
