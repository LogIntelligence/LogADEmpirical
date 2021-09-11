import json
from collections import Counter
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from logadempirical.PLELog.data.Instance import *


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r = list(['0']) * (n - len(r)) + r
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for _ in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        sample_logs.append(logs[random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


# https://stackoverflow.com/questions/15357422/python-determine-if-a-string-should-be-converted-into-int-or-float
def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def load_features(data_path, only_normal=True, min_len=0):
    print(data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    if only_normal:
        logs = []
        for seq in data:
            if len(seq['EventId']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = max(seq['Label'].tolist())
            else:
                label = seq['Label']
            if label == 0:
                logs.append((seq['EventId'], label))
    else:
        logs = []
        no_abnormal = 0
        for seq in data:
            if len(seq['EventId']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = seq['Label'].tolist()
                if max(label) > 0:
                    no_abnormal += 1
            else:
                label = seq['Label']
                if label > 0:
                    no_abnormal += 1
            logs.append((seq['EventId'], label))
        print("Number of abnormal sessions:", no_abnormal)
    return logs

def sliding_window_test(data_iter, window_size=10, is_train=True, id2temp={}, idx=0):

    # event2semantic_vec = read_json("dataset/BGL/embeddings.json")
    result_logs = []
    labels = []

    num_sessions = 0
    num_seq = idx
    duplicate_seq = {}
    id2temp["padding"] = ""
    for idx, (line, lbls) in enumerate(data_iter):
        line = list(line)
        if (num_sessions + 1) % 100 == 0:
            print("processed %s lines" % (num_sessions + 1))

        seq_len = max(window_size, len(line))
        line = ["padding"] * (seq_len - len(line)) + line
        for i in range(len(line) - window_size + 1):
            if not isinstance(lbls, int):
                label = max(lbls[i: i + window_size])
            else:
                label = lbls

            seq = line[i: i + window_size]
            seq = " ".join(seq)
            if seq in duplicate_seq.keys():
                pos = duplicate_seq[seq]
                if labels[pos] != label:
                    labels[pos] = 0
                    result_logs[pos].type = "Normal"
                    result_logs[pos].tag = "no"
                continue

            duplicate_seq[seq] = len(labels)
            sequential_pattern = line[i:i + window_size]
            sequential_pattern = [id2temp[x] for x in sequential_pattern]
            inst = parseInstance(sequential_pattern, num_seq, "Anomaly" if label == 1 else "Normal")
            result_logs.append(inst)
            labels.append(label)
            num_seq += 1
        num_sessions += 1

    if is_train:
        print('number of sessions {}'.format(num_sessions))
        print('number of seqs {}'.format(len(result_logs)), len(duplicate_seq.keys()))
    return result_logs, labels, num_seq


def sliding_window(data_iter, window_size=10, is_train=True, id2temp={}, idx=0):

    # event2semantic_vec = read_json("dataset/BGL/embeddings.json")
    result_logs = []
    labels = []

    num_sessions = 0
    num_seq = idx
    id2temp["padding"] = ""
    for idx, (line, lbls) in enumerate(data_iter):
        line = list(line)
        if (num_sessions + 1) % 100 == 0:
            print("processed %s lines" % (num_sessions + 1))

        if not isinstance(lbls, int):
            label = max(lbls)
        else:
            label = lbls

        sequential_pattern = line
        sequential_pattern = [id2temp[x] for x in sequential_pattern]
        inst = parseInstance(sequential_pattern, num_seq, "Anomaly" if label == 1 else "Normal")
        result_logs.append(inst)
        labels.append(label)
        num_seq += 1
        num_sessions += 1

    if is_train:
        print('number of sessions {}'.format(num_sessions))
        print('number of seqs {}'.format(len(result_logs)))
    return result_logs, labels, num_seq
