import json
from collections import Counter
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

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


def split_features(data_path, train_ratio=1, scale=None, scale_path=None, min_len=0):
    with open(data_path, 'r') as f:
        data = f.readlines()

    sample_size = int(len(data) * train_ratio)
    data = data[:sample_size]
    logkeys = []
    times = []
    for line in data:
        line = [ln.split(",") for ln in line.split()]

        if len(line) < min_len:
            continue

        line = np.array(line)
        # if time duration exists in data
        if line.shape[1] == 2:
            tim = line[:, 1].astype(float)
            tim[0] = 0
            logkey = line[:, 0]
        else:
            logkey = line.squeeze()
            # if time duration doesn't exist, then create a zero array for time
            tim = np.zeros(logkey.shape)

        logkeys.append(logkey.tolist())
        times.append(tim.tolist())

    if scale is not None:
        total_times = np.concatenate(times, axis=0).reshape(-1,1)
        scale.fit(total_times)

        for i, tn in enumerate(times):
            tn = np.array(tn).reshape(-1,1)
            times[i] = scale.transform(tn).reshape(-1).tolist()

        with open(scale_path, 'wb') as f:
            pickle.dump(scale, f)
        print("Save scale {} at {}\n".format(scale, scale_path))

    return logkeys, times


def sliding_window(data_iter, vocab, window_size, is_train=True):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    #event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    result_logs['Parameters'] = []
    labels = []

    num_sessions = 0
    num_classes = len(vocab)

    duplicate_seq = {}

    for line, params in zip(*data_iter):
        if num_sessions % 1000 == 0:
            print("processed %s lines"%num_sessions, end='\r')
        num_sessions += 1

        line = [vocab.stoi.get(ln, vocab.unk_index) for ln in line]

        session_len = max(len(line), window_size) + 1# predict the next one
        padding_size = session_len - len(line)
        params = params + [0] * padding_size
        line = line + [vocab.pad_index] * padding_size

        for i in range(session_len - window_size):
            seq = line[i: i + window_size + 1]
            seq = map(lambda k: str(k), seq)
            seq = " ".join(seq)
            if seq in duplicate_seq.keys():
                continue

            duplicate_seq[seq] = 1
            Parameter_pattern = params[i:i + window_size]
            Sequential_pattern = line[i:i + window_size]
            Semantic_pattern = []

            Quantitative_pattern = [0] * num_classes
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]

            Sequential_pattern = np.array(Sequential_pattern)
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
            Parameter_pattern = np.array(Parameter_pattern)[:, np.newaxis] #increase one more dimension

            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
            result_logs['Semantics'].append(Semantic_pattern)
            result_logs["Parameters"].append(Parameter_pattern)
            labels.append([line[i + window_size], params[i + window_size]])

    if is_train:
        print('number of sessions {}'.format(num_sessions))
        print('number of seqs {}'.format(len(result_logs['Sequentials'])))

    return result_logs, labels


def session_window(data_dir, datatype, train_ratio=1, e_name="", num_events=29):
    event2semantic_vec = read_json(os.path.join(data_dir, e_name))
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += 'train'
        f_normal = data_dir
        f_abnormal = data_dir + "_abnormal"
    elif datatype == 'test':
        data_dir += 'test'
        f_normal = data_dir + "_normal"
        f_abnormal = data_dir + "_abnormal"
    else:
        raise NotImplementedError
    x, y = [], []
    with open(f_normal, mode="r") as f:
        for line in f.readlines():
            line = line.strip()
            x.append(line.split())
            y.append(0)

    with open(f_abnormal, mode="r") as f:
        for line in f.readlines():
            line = line.strip()
            x.append(line.split())
            y.append(1)

    for i, line in enumerate(x):
        Sequential_pattern = trp(line, 100)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == "0":
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[event])
        Quantitative_pattern = [0] * num_events
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern) # [:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(y[i])

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))

    if train_ratio != 1:
        (train_x, train_y), (val_x, val_y) = train_test_split(result_logs, labels, test_size=train_ratio)
        return (train_x, train_y), (val_x, val_y)
    else:
        return result_logs, labels
