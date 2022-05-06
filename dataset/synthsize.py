import os
import pickle
from copy import copy, deepcopy

import pandas as pd
import numpy as np
import torch
from nltk.corpus import stopwords
from tqdm import tqdm
import re

stop_words = set(stopwords.words('english'))
word_list = ['process', "thread", "block", "restart", "network", "response", "get", "post", "warning", "bgl",
             "update", "data", "transform", "push", "free", "pointer", "new", "send", "disk", "hardware",
             "change", "read", "write", "lock", "cash", "print", "value", "ping", "route", "delete", "catch",
             "try", "connect", "compute", "dead", "finish", "gate", "IP", "application", "compile", "over",
             "stack", "sort", "argument", "chunk", "segment", "import", "register", "convert", "future", "code",
             "memory", "share", "auto", "bus", "type", "drive", "copy", "item", "mode", "contain", "default"]

temp_df = pd.read_csv("./BGL.log_templates.csv")
temp_map = {}
for idx, row in temp_df.iterrows():
    temp_map[row['EventId']] = row['EventTemplate']


def clean_template(template, remove_stop_words=False):
    word_tokens = template.split()
    if remove_stop_words:  # remove stop words, we can close this function
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    else:
        filtered_sentence = [w.lower() for w in word_tokens]
    return " ".join(filtered_sentence)


def generate_new_log(log, change_ratio):
    log = clean_template(log, remove_stop_words=True)
    words = log.split()
    num_tokens = len(words)
    change_num = int(num_tokens * change_ratio) + 1
    add_word_num = np.random.randint(max(1, change_num - 1), change_num + 1, 1)
    noise_indices = torch.randperm(num_tokens + add_word_num[0] - 1)[:add_word_num[0]] + 1
    noise_mask = torch.zeros(size=(num_tokens + add_word_num[0],), dtype=torch.bool)
    noise_mask[noise_indices] = 1
    noise_mask = noise_mask.detach().tolist()
    add_words = np.random.choice(word_list, add_word_num, replace=True)
    add_words = add_words.tolist()
    new_logs = []
    for mask in noise_mask:
        if mask:
            new_logs.append(add_words.pop(0))
        else:
            new_logs.append(words.pop(0))
    return " ".join(new_logs)


def gen_new_templates(t_set, ratio):
    for k, v in t_set.items():
        t_set[k]['new'] = generate_new_log(v['orig'], ratio)
    return t_set


def extract(s, t):
    if "<*>" not in t:
        return []
    template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', t)
    template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
    template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
    rg = re.compile(template_regex)
    # print(rg, s)
    parameter_list = rg.findall(s)  # re.findall(template_regex, s)
    parameter_list = parameter_list[0] if parameter_list else ()
    parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
    return parameter_list

def new_content(c, temp):
    v = extract(c, temp['orig'])
    l = temp['new']
    while len(v) > 0:
        u = v.pop(0)
        l = l.replace('<*>', u, 1)
    return l


def create_synthetic_dataset(data, temps):
    for i in range(len(data)):
        e_seq = data[i]['EventId']
        c_seq = data[i]['Seq'].tolist()
        new_seq = []
        for j in range(len(c_seq)):
            new_seq.append(new_content(c_seq[j], temps[e_seq[j]]))
        data[i]['Seq'] = new_seq
    return data


if __name__ == '__main__':
    df = pd.read_csv("BGL.log_templates.csv")
    temp_map = {}
    for idx, row in df.iterrows():
        if row['EventId'] not in temp_map.keys():
            temp_map[row['EventId']] = {}
            temp_map[row['EventId']]['orig'] = row['EventTemplate']
    print("Loading...")
    with open("train.pkl", "rb") as f:
        train = pickle.load(f)
    with open("test.pkl", "rb") as f:
        test = pickle.load(f)
    print("Loaded!")
    for r in [0.05, 0.1, 0.15, 0.2, 0.3]:
        print(r)
        templates = deepcopy(temp_map)
        templates = gen_new_templates(templates, ratio=r)
        te = create_synthetic_dataset(deepcopy(test), templates)
        with open("synthetic/test_{0}.pkl".format(r), "wb") as f:
            pickle.dump(te, f)
        with open("synthetic/templates_{0}.pkl".format(r), "wb") as f:
            pickle.dump(templates, f)
