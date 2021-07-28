import json
from collections import Counter
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
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
        for seq in data:
            if len(seq['EventId']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = seq['Label'].tolist()
            else:
                label = seq['Label']
            logs.append((seq['EventId'], label))
    return logs


def sliding_window(data_iter, vocab, window_size, is_train=True, data_dir="dataset/", is_predict_logkey=True,
                   e_name="embeddings.json"):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    event2semantic_vec = read_json(os.path.join(data_dir, e_name))
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    result_logs['Parameters'] = []
    result_logs['idx'] = []
    labels = []

    num_sessions = 0
    num_classes = len(vocab)

    duplicate_seq = {}

    n_all = 0
    for (line, lbls) in data_iter:
        if (num_sessions + 1) % 100 == 0:
            print("processed %s lines" % (num_sessions + 1), end='\r')
        orig_line = line.copy()
        line = [vocab.stoi.get(ln, vocab.unk_index) for ln in line]
        if len(line) < window_size + 1:
            continue
        session_len = len(line)  # , window_size) + 1  # predict the next one
        # padding_size = session_len - len(line)
        # orig_line = orig_line + ["0"] * padding_size
        # line = line + [vocab.pad_index] * padding_size
        if not is_train:
            duplicate_seq = {}

        for i in range(session_len - window_size):
            if is_predict_logkey:
                label = line[i + window_size]
            else:
                if isinstance(lbls, list):
                    label = max(lbls[i: i + window_size])
                else:
                    label = lbls
            n_all += 1
            seq = line[i: i + window_size].copy()
            if is_predict_logkey:
                seq.append(line[i + window_size])
            else:
                seq.append(label)
            seq = map(lambda k: str(k), seq)
            seq = " ".join(seq)
            if seq in duplicate_seq.keys():
                continue

            duplicate_seq[seq] = 1
            Sequential_pattern = line[i:i + window_size].copy()
            Semantic_pattern = []
            for event in orig_line[i:i + window_size].copy():
                if event == "0":
                    Semantic_pattern.append([-1] * 300)
                else:
                    Semantic_pattern.append(event2semantic_vec[event])

            Quantitative_pattern = [0] * num_classes
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]

            Sequential_pattern = np.array(Sequential_pattern)
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
            result_logs['Semantics'].append(Semantic_pattern)
            result_logs['idx'].append(num_sessions)

            # if label == 1:
            #     print(Sequential_pattern)
            labels.append(label)
        num_sessions += 1

    # if is_train:
    print('number of sessions {}'.format(num_sessions))
    print('number of seqs {}'.format(len(result_logs['Sequentials'])))
    result_logs['Sequentials'], result_logs['Quantitatives'], result_logs['Semantics'], result_logs[
        'idx'], labels = shuffle(result_logs['Sequentials'], result_logs['Quantitatives'], result_logs['Semantics'],
                                 result_logs['idx'], labels)
    return result_logs, labels
