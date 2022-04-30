import pandas as pd
import numpy as np
from collections import OrderedDict
import re

from sklearn.utils import shuffle
import pickle
import os

from modules.data.Instance import parse_instance


def load_fixed_windows_instances(log_file, train_ratio=0.2, window_size=20):
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
    while i < n_train - window_size:
        c += 1
        if c % 1000 == 0:
            print("Loading {0:.2f}".format(i * 100 / n_train))
        event_seq, eventid_seq, content_seq = [], [], []
        label = 'normal'
        for j in range(i, i + window_size):
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
        i = i + window_size
    print("last train index:", i)

    x_te = []
    for i in range(n_train, len(logs) - window_size, window_size):
        if i % 1000 == 0:
            print("Loading {:.2f}".format(i * 100 / n_train))

        event_seq, eventid_seq, content_seq = [], [], []
        label = 'normal'
        for j in range(i, i + window_size):
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


def load_sliding_windows_instances(log_file, train_ratio=0.5, window_size=20, step_size=1):
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
    while i < n_train - window_size:
        c += 1
        if c % 1000 == 0:
            print("Loading {0:.2f}".format(i * 100 / n_train))
        event_seq, eventid_seq, content_seq = [], [], []
        label = 'normal'
        for j in range(i, i + window_size):
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
    for i in range(n_train, len(logs) - window_size, step_size):
        if i % 1000 == 0:
            print("Loading {:.2f}".format(i * 100 / n_train))

        event_seq, eventid_seq, content_seq = [], [], []
        label = 'normal'
        for j in range(i, i + window_size):
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


import json
import datetime


def timestamp(d, t):
    s = d + " " + t[:-4]
    dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


def load_Hadoop(log_file, label_file, train_ratio=0.5, window_size=20):
    df = pd.read_csv(log_file)
    structured_logs = df.to_dict("records")
    with open(label_file, mode="r", encoding="utf8") as f:
        label = json.load(f)
    x = []
    for sid, session in label.items():
        i = session['start']
        while i < session['end']:
            l = i
            r = i
            e_time = timestamp(structured_logs[i]['Date'], structured_logs[i]['Time']) + 3600
            for j in range(l, session['end']):
                if timestamp(structured_logs[j]['Date'], structured_logs[j]['Time']) >= e_time:
                    break
                r = j + 1
            event_ids = []
            events = []
            contents = []
            label = session['label'] == "Anomaly"
            print(l, r)
            for j in range(l, r):
                event_id = structured_logs[j]['EventId']
                event = structured_logs[j]['EventTemplate']
                content = structured_logs[j]['Content']
                event_ids.append(event_id)
                events.append(event)
                contents.append(content)

            x.append({"SessionId": len(x), "EventId": events.copy(), "Label": label})
                # parse_instance(events.copy(), None, label, event_ids=event_ids.copy(),
                #                     messages=contents.copy(),
                #                     confidence=1))
            i = r

    x = shuffle(x)
    n_train = int(len(x) * train_ratio)
    x_tr = x[:n_train]
    x_te = x[n_train:]

    num_total = len([instance["Label"] for instance in x])
    num_pos = sum([instance["Label"] for instance in x])

    num_train = len([instance["Label"] for instance in x_tr])
    num_train_pos = sum([instance["Label"] for instance in x_tr])

    num_test = len([instance["Label"] for instance in x_te])
    num_test_pos = sum([instance["Label"] for instance in x_te])

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))
    return x_tr, x_te


if __name__ == '__main__':

    # x_tr, x_te = load_Hadoop("spirit/Hadoop.log_structured.csv", "hadoop/label.json", train_ratio=0.8)

    x_tr, x_te = load_fixed_windows_instances(
        "../../LogVectorization/logparser/benchmark/Drain_result/Spirit5M.log_structured.csv",
        train_ratio=0.8,
        window_size=20)

    # with open("hadoop/train.pkl", mode="wb") as f:
    #     pickle.dump(x_tr, f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open("hadoop/test.pkl", mode="wb") as f:
    #     pickle.dump(x_te, f, protocol=pickle.HIGHEST_PROTOCOL)
