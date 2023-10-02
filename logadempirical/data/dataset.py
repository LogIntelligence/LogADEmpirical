# -*- coding: utf-8 -*-
import pdb

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
from logadempirical.data.vocab import Vocab
from typing import List, Tuple, Optional, Any


class BaseDataset(Dataset):
    def __init__(self,
                 sequentials: Optional[List[List[int]]] = None,
                 quantitatives: Optional[List[List[float]]] = None,
                 semantics: Optional[List[List[float]]] = None,
                 is_unsupervised: bool = True,
                 labels: Optional[List[int]] = None,
                 idxs: Optional[List[int]] = None,
                 remove_duplicates: bool = True):
        """ Base Dataset class for log data
        Parameters
        ----------
        sequentials: List of log sequences
        quantitatives: List of quantitative features
        semantics: List of semantic features
        is_unsupervised: unsupervised or supervised
        labels: List of labels
        idxs: List of indexes
        remove_duplicates: remove duplicates from data
        """
        self.sequentials = sequentials
        self.quantitatives = quantitatives
        self.semantics = semantics
        self.is_unsupervised = is_unsupervised
        self.labels = labels
        self.idxs = idxs
        self.weights = []
        if remove_duplicates:
            if self.sequentials is None:
                raise ValueError('Provide sequentials to remove duplicates')
            self.remove_duplicates()

    def remove_duplicates(self):
        self.remove_duplicates_from_data(self.sequentials)

    def remove_duplicates_from_data(self, data):
        """
        remove duplicates from data
        :param data:
        :return:
        """
        unique_data = {}
        occurrences = defaultdict(int)
        unique_labels = defaultdict(int)
        for idx, (seq, label) in enumerate(zip(data, self.labels)):
            if self.is_unsupervised:
                sample = tuple(seq + [label])
            else:
                sample = tuple(seq)
            if sample not in unique_data:
                unique_data[sample] = idx
                occurrences[sample] += 1
                unique_labels[sample] = label
            else:
                occurrences[sample] += 1
                if not self.is_unsupervised:
                    unique_labels[sample] = min(unique_labels[sample], label)
        if not self.is_unsupervised:
            self.labels = [unique_labels[tuple(seq)] for seq in data]
        filtered_idxs = sorted(unique_data.values())
        n_data = len(self.sequentials)
        for idx in filtered_idxs:
            if self.is_unsupervised:
                sample = tuple(data[idx] + [self.labels[idx]])
            else:
                sample = tuple(data[idx])
            self.weights.append(occurrences[sample] / n_data)
        self.sequentials = [data[idx] for idx in filtered_idxs]
        if self.quantitatives is not None:
            self.quantitatives = [self.quantitatives[idx] for idx in filtered_idxs]
        if self.semantics is not None:
            self.semantics = [self.semantics[idx] for idx in filtered_idxs]
        self.labels = [self.labels[idx] for idx in filtered_idxs]
        if self.idxs is not None:
            self.idxs = [self.idxs[idx] for idx in filtered_idxs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raise NotImplementedError

    def collate_fn(self, batch):
        raise NotImplementedError


class LogDataset(BaseDataset):
    def __init__(self, sequentials=None, quantitatives=None, semantics=None, is_unsupervised=False, labels=None,
                 idxs=None, remove_duplicates=True):
        super(LogDataset, self).__init__(sequentials=sequentials, quantitatives=quantitatives, semantics=semantics,
                                         is_unsupervised=is_unsupervised, labels=labels, idxs=idxs,
                                         remove_duplicates=remove_duplicates)
        if self.sequentials is None and self.quantitatives is None and self.semantics is None:
            raise ValueError('Provide at least one feature type')

    def __getitem__(self, idx):
        try:
            item = {'label': self.labels[idx], 'idx': self.idxs[idx]}
        except:
            pdb.set_trace()
        if self.sequentials is not None:
            item['sequential'] = torch.from_numpy(np.array(self.sequentials[idx]))
        if self.quantitatives is not None:
            item['quantitative'] = torch.from_numpy(np.array(self.quantitatives[idx], )[:, np.newaxis]).float()
        if self.semantics is not None:
            item['semantic'] = torch.from_numpy(np.array(self.semantics[idx])).float()

        return item

    # def collate_fn(self, batch, feature_name='semantic', padding_side="right"):
    #     # pdb.set_trace()
    #     return batch
    #     max_length = max([len(b[feature_name]) for b in batch])
    #     dimension = {k: batch[0][k][0].shape[0] for k in batch[0].keys() if k != 'label' and batch[0][k] is not None}
    #     if padding_side == "right":
    #         padded_batch = []
    #         for b in batch:
    #             sample = {}
    #             for k, v in b.items():
    #                 if k == 'label':
    #                     sample[k] = v
    #                 elif v is None:
    #                     sample[k] = None
    #                 else:
    #                     sample[k] = torch.from_numpy(
    #                         np.array(v + [np.zeros(dimension[k], )] * (max_length - len(v))))
    #             padded_batch.append(sample)
    #     elif padding_side == "left":
    #         padded_batch = []
    #         for b in batch:
    #             sample = {}
    #             for k, v in b.items():
    #                 if k == 'label':
    #                     sample[k] = v
    #                 elif v is None:
    #                     sample[k] = None
    #                 else:
    #                     sample[k] = torch.from_numpy(
    #                         np.array([np.zeros(dimension[k], )] * (max_length - len(v)) + v))
    #             padded_batch.append(sample)
    #     else:
    #         raise ValueError("padding_side should be either 'right' or 'left'")
    #
    #     # convert to tensor
    #     padded_batch = {
    #         k: torch.stack([sample[k] for sample in padded_batch])
    #         for k in padded_batch[0].keys()
    #     }
    #     return padded_batch


class MaskedDataset(BaseDataset):
    def __init__(self, sequentials, vocab, seq_len, encoding="utf-8", on_memory=True,
                 predict_mode=False, mask_ratio=0.75, idx=None):
        """

        :param sequentials: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        super(MaskedDataset, self).__init__(sequentials=sequentials, labels=None, idxs=idx)
        self.vocab = vocab
        self.seq_len = seq_len
        self.idxs = idx
        self.on_memory = on_memory
        self.encoding = encoding
        self.predict_mode = predict_mode
        self.mask_ratio = mask_ratio
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4

    def __len__(self):
        return len(self.sequentials)

    def __getitem__(self, idx):
        seq = self.sequentials[idx]

        masked, label = self.random_item(list(seq))

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        seq = [self.sos_index] + masked
        label = [self.pad_index] + label

        return seq, idx, label

    def random_item(self, tokens):
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

        output = {"sequential": [], "label": [], "idx": []}
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

# def data_collate(batch, feature_name='semantic', padding_side="right"):
#     max_length = max([len(b[feature_name]) for b in batch])
#     dimension = {k: batch[0][k][0].shape[0] for k in batch[0].keys() if k != 'label' and batch[0][k] is not None}
#     if padding_side == "right":
#         padded_batch = []
#         for b in batch:
#             sample = {}
#             for k, v in b.items():
#                 if k == 'label':
#                     sample[k] = v
#                 elif v is None:
#                     sample[k] = None
#                 else:
#                     sample[k] = torch.from_numpy(
#                         np.array(v + [np.zeros(dimension[k], )] * (max_length - len(v))))
#             padded_batch.append(sample)
#     elif padding_side == "left":
#         padded_batch = []
#         for b in batch:
#             sample = {}
#             for k, v in b.items():
#                 if k == 'label':
#                     sample[k] = v
#                 elif v is None:
#                     sample[k] = None
#                 else:
#                     sample[k] = torch.from_numpy(
#                         np.array([np.zeros(dimension[k], )] * (max_length - len(v)) + v))
#             padded_batch.append(sample)
#     else:
#         raise ValueError("padding_side should be either 'right' or 'left'")
#
#     # convert to tensor
#     padded_batch = {
#         k: torch.stack([sample[k] for sample in padded_batch])
#         for k in padded_batch[0].keys()
#     }
#     return padded_batch
