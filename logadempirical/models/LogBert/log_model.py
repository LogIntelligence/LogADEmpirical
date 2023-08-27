import pdb

import torch.nn as nn
import torch
from .bert import BERT
from typing import Optional
from logadempirical.models.utils import ModelOutput
from torch.nn import LogSoftmax


class BERTLog(nn.Module):
    """
    BERT Log Model
    """

    def __init__(self, bert: BERT, vocab_size, criterion: Optional[nn.Module] = None, hidden_size: int = 128,
                 n_class: int = 1, is_bilstm: bool = True):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLogModel(self.bert.hidden, vocab_size)
        self.time_lm = TimeLogModel(self.bert.hidden)
        # self.fnn_cls = LinearCLS(self.bert.hidden)
        # self.cls_lm = LogClassifier(self.bert.hidden)
        self.fc = nn.Linear(self.bert.hidden, vocab_size)
        self.num_directions = 2 if is_bilstm else 1
        self.fc2 = nn.Linear(vocab_size, n_class)  # sửa đổi so với gốc
        self.criterion = nn.NLLLoss()
        # self.result = {"logkey_output": None, "cls_output": None, }

    def forward(self, batch, time_info=None, device="cpu"):
        x = batch["sequential"]
        try:
            y = batch['label']
            y = y
        except KeyError:
            y = None
        x = self.bert(x, time_info=time_info)
        x = self.mask_lm(x)
        logits = self.fc2(x)
        probabilities = torch.softmax(x, dim=-1)

        # self.result["logkey_output"] = self.mask_lm(x)

        # self.result["cls_output"] = x[:, 0]
        loss = None
        # logits = logits.view(-1).type(torch.FloatTensor)
        # y = y.view(-1).type(torch.FloatTensor)
        # pdb.set_trace()
        if y is not None and self.criterion is not None:
            loss = self.criterion(x.transpose(1, 2).type(torch.FloatTensor), y.type(torch.LongTensor))
        return ModelOutput(logits=logits, probabilities=probabilities, loss=loss, embeddings=x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, src, device="cpu"):
        del src['label']
        return self.forward(src, device=device).probabilities

    def predict_class(self, src, top_k=1, device="cpu"):
        del src['label']
        return torch.topk(self.forward(src, device=device).probabilities, k=top_k, dim=1).indices


class MaskedLogModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class TimeLogModel(nn.Module):
    def __init__(self, hidden, time_size=1):
        super().__init__()
        self.linear = nn.Linear(hidden, time_size)

    def forward(self, x):
        return self.linear(x)


class LogClassifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, cls):
        return self.linear(cls)


class LinearCLS(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.linear(x)
