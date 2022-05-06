import json
from collections import Counter
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from logadempirical.logdeep.dataset import bert_encoder


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
                try:
                    logs.append((seq['EventId'], label, seq['Seq'].tolist()))
                except:
                    logs.append((seq['EventId'], label, seq['Seq']))
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
            try:
                logs.append((seq['EventId'], label, seq['Seq'].tolist()))
            except:
                logs.append((seq['EventId'], label, seq['Seq']))
        print("Number of abnormal sessions:", no_abnormal)
    return logs


def sliding_window(data_iter, vocab, window_size, is_train=True, data_dir="dataset/", is_predict_logkey=True,
                   e_name="embeddings.json", semantics=True, sample_ratio=1, in_size=768):
    if e_name == "neural":
        event2semantic_vec = {}
        is_bert = True
    else:
        event2semantic_vec = read_json(os.path.join(data_dir, e_name))
        is_bert = False
    result_logs = []
    labels = []

    num_sessions = 0
    num_classes = len(vocab)

    duplicate_seq = {}
    for idx, (orig_line, lbls, contents) in enumerate(data_iter):
        orig_line = list(orig_line)
        # print(len(orig_line))
        # print(orig_line[:window_size])
        if (num_sessions + 1) % 100 == 0:
            print("processed %s lines" % (num_sessions + 1), end='\r')
        line = [vocab.stoi.get(ln, vocab.find_similar(ln)) for ln in orig_line]
        if is_predict_logkey:
            seq_len = max(window_size + 1, len(line))
        else:
            seq_len = max(window_size, len(line))
        line = [vocab.pad_index] * (seq_len - len(line)) + line
        orig_line = ["padding"] * (seq_len - len(orig_line)) + orig_line
        # print(contents)
        contents = ["padding"] * (seq_len - len(contents)) + contents
        # if not is_train:
        #     duplicate_seq = {}
        for i in range(len(line) - window_size if is_predict_logkey else len(line) - window_size + 1):
            if is_predict_logkey:
                if i + window_size >= len(line) and is_predict_logkey:
                    break
                label = line[i + window_size]
            else:
                if not isinstance(lbls, int):
                    label = max(lbls[i: i + window_size])
                else:
                    label = lbls

            if is_bert:
                seq = contents[i: i + window_size]
            else:
                seq = line[i: i + window_size]
            if is_predict_logkey:
                seq.append(label)
            seq = list(map(lambda k: str(k), seq))
            seq = " ".join(seq)
            # if seq in duplicate_seq.keys():
            #     if not is_predict_logkey:
            #         pos = duplicate_seq[seq]
            #         if labels[pos] != label:
            #             labels[pos] = 0
            #     continue

            # duplicate_seq[seq] = len(labels)
            sequential_pattern = []#line[i:i + window_size]
            semantic_pattern = []
            if semantics:
                if is_bert:
                    seq_logs = contents[i: i + window_size]
                else:
                    seq_logs = orig_line[i: i + window_size]
                for event in seq_logs:
                    if event == "padding":
                        semantic_pattern.append([-1] * in_size)
                    else:
                        if is_bert:
                            semantic_pattern.append(bert_encoder(event, event2semantic_vec))
                        else:
                            semantic_pattern.append(event2semantic_vec[event])

            quantitative_pattern = [0]# * num_classes
            # log_counter = Counter(sequential_pattern)
            #
            # for key in log_counter:
            #     try:
            #         quantitative_pattern[key] = log_counter[key]
            #     except:
            #         pass  # ignore unseen events or padding key

            sequential_pattern = np.array(sequential_pattern)
            quantitative_pattern = np.array(quantitative_pattern)[:, np.newaxis]

            result_logs.append(([sequential_pattern, quantitative_pattern, semantic_pattern], idx))
            labels.append(label)
        num_sessions += 1

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)
    if is_train:
        print('number of sessions {}'.format(num_sessions))
        print('number of seqs {}'.format(len(result_logs)))
    return result_logs, labels
