import pandas as pd
import numpy as np
from collections import OrderedDict
import re

from sklearn.utils import shuffle
import pickle
import os

from modules.data.Instance import parse_instance


def load_fixed_windows_instances(log_file, train_ratio=0.5, windows_size=20):
    print("Loading", log_file)
    logs = pd.read_csv(log_file, memory_map=True)
    logs = logs.to_dict('records')
    print("Loaded")
    x_tr, y_tr = [], []
    i = 0
    failure_count = 0
    n_train = int(len(logs) * train_ratio)
    c = 0
    num_train_pos, num_test_pos = 0, 0
    while i < n_train - windows_size:
        c += 1
        if c % 1000 == 0:
            print("Loading {0:.2f}".format(i * 100 / n_train))
        event_seq, eventid_seq, content_seq = [], [], []
        label = 'normal'
        for j in range(i, i + windows_size):
            if logs[j]['Label'] != "-":
                label = 'anomaly'
                failure_count += 1
            content = logs[j]['Content']
            event = logs[j]['EventTemplate']
            event_id = logs[j]['EventId']
            event_seq.append(event)
            eventid_seq.append(event_id)
            content_seq.append(content)
        if label == 'anomaly':
            num_train_pos += 1
        x_tr.append(parse_instance(event_seq.copy(), None, label, event_ids=eventid_seq.copy(),
                                   messages=content_seq.copy(),
                                   confidence=1))
        i = i + windows_size
    print("last train index:", i)

    x_te = []
    for i in range(n_train, len(logs) - windows_size, windows_size):
        if i % 1000 == 0:
            print("Loading {:.2f}".format(i * 100 / n_train))

        event_seq, eventid_seq, content_seq = [], [], []
        label = 'normal'
        for j in range(i, i + windows_size):
            if logs[j]['Label'] != "-":
                label = 'anomaly'
                failure_count += 1
            content = logs[j]['Content']
            event = logs[j]['EventTemplate']
            event_id = logs[j]['EventId']
            event_seq.append(event)
            eventid_seq.append(event_id)
            content_seq.append(content)
        if label == 'anomaly':
            num_test_pos += 1
        x_te.append(parse_instance(event_seq.copy(), None, label, event_ids=eventid_seq.copy(),
                                   messages=content_seq.copy(),
                                   confidence=1))

    print("Total failure logs: {0}".format(failure_count))

    num_train = len(x_tr)
    num_test = len(x_te)
    num_total = num_train + num_test
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return x_tr, x_te

def load_sliding_windows_instances(log_file, train_ratio=0.5, windows_size=20, step_size=1):
    print("Loading", log_file)
    logs = pd.read_csv(log_file, memory_map=True)
    logs = logs.to_dict('records')
    print("Loaded")
    x_tr, y_tr = [], []
    i = 0
    failure_count = 0
    n_train = int(len(logs) * train_ratio)
    c = 0
    num_train_pos, num_test_pos = 0, 0
    while i < n_train - windows_size:
        c += 1
        if c % 1000 == 0:
            print("Loading {0:.2f}".format(i * 100 / n_train))
        event_seq, eventid_seq, content_seq = [], [], []
        label = 'normal'
        for j in range(i, i + windows_size):
            if logs[j]['Label'] != "-":
                label = 'anomaly'
                failure_count += 1
            content = logs[j]['Content']
            event = logs[j]['EventTemplate']
            event_id = logs[j]['EventId']
            event_seq.append(event)
            eventid_seq.append(event_id)
            content_seq.append(content)
        if label == 'anomaly':
            num_train_pos += 1
        x_tr.append(parse_instance(event_seq.copy(), None, label, event_ids=eventid_seq.copy(),
                                   messages=content_seq.copy(),
                                   confidence=1))
        i = i + step_size
    print("last train index:", i)

    x_te = []
    for i in range(n_train, len(logs) - windows_size, step_size):
        if i % 1000 == 0:
            print("Loading {:.2f}".format(i * 100 / n_train))

        event_seq, eventid_seq, content_seq = [], [], []
        label = 'normal'
        for j in range(i, i + windows_size):
            if logs[j]['Label'] != "-":
                label = 'anomaly'
                failure_count += 1
            content = logs[j]['Content']
            event = logs[j]['EventTemplate']
            event_id = logs[j]['EventId']
            event_seq.append(event)
            eventid_seq.append(event_id)
            content_seq.append(content)
        if label == 'anomaly':
            num_test_pos += 1
        x_te.append(parse_instance(event_seq.copy(), None, label, event_ids=eventid_seq.copy(),
                                   messages=content_seq.copy(),
                                   confidence=1))

    print("Total failure logs: {0}".format(failure_count))

    num_train = len(x_tr)
    num_test = len(x_te)
    num_total = num_train + num_test
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return x_tr, x_te

if __name__ == '__main__':
    x_tr, x_te = load_fixed_windows_instances("bgl/BGL.log_structured.csv", train_ratio=0.8,
                                              windows_size=100)

    with open("./bgl/train_fixed100_instances.pkl", mode="wb") as f:
        pickle.dump(x_tr, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("./bgl/test_fixed100_instances.pkl", mode="wb") as f:
        pickle.dump(x_te, f, protocol=pickle.HIGHEST_PROTOCOL)
