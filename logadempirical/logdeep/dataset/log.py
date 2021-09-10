#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class log_dataset(Dataset):
    def __init__(self, logs, labels):
        self.logs = []
        for i in range(len(labels)):
            features = [torch.tensor(logs[i][0][0], dtype=torch.long)]
            for j in range(1, len(logs[i][0])):
                features.append(torch.tensor(logs[i][0][j], dtype=torch.float))
            self.logs.append({
                "features": features,
                "idx": logs[i][1]
            })
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.logs[idx], self.labels[idx]


if __name__ == '__main__':
    data_dir = '../../data/'
