#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pickle
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import auc

from logbert.logdeep.dataset.log import log_dataset
from logbert.logdeep.dataset.sample import sliding_window
from logbert.logdeep.models.lstm import deeplog, loganomaly, robustlog
from logbert.logdeep.models.cnn import TextCNN
from logbert.neural_log.transformers import TransformerClassification


def generate(output_dir, name):
    print("Loading", output_dir + name)
    with open(output_dir + name, 'rb') as f:
        data_iter = pickle.load(f)
    normal_iter = []
    abnormal_iter = []
    for seq in data_iter:
        if not isinstance(seq['Label'], int):
            label = max(seq['Label'].tolist())
        else:
            label = seq['Label']
        if label == 0:
            normal_iter.append((seq['EventId'], label))
        else:
            abnormal_iter.append((seq['EventId'], label))
    return normal_iter, abnormal_iter


class Predicter():
    def __init__(self, options):
        self.data_dir = options['data_dir']
        self.output_dir = options['output_dir']
        self.model_dir = options['model_dir']
        self.vocab_path = options["vocab_path"]
        self.model_path = options['model_path']
        self.model_name = options['model_name']

        self.device = options['device']
        self.window_size = options['window_size']
        self.min_len = options["min_len"]
        self.seq_len = options["seq_len"]
        self.history_size = options['history_size']

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.test_ratio = options["test_ratio"]
        self.num_candidates = options['num_candidates']

        self.input_size = options["input_size"]
        self.hidden_size = options["hidden_size"]
        self.embedding_dim = options["embedding_dim"]
        self.num_layers = options["num_layers"]

        self.batch_size = options['batch_size']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.embeddings = options['embeddings']

        # transformers' parameters
        self.num_encoder_layers = options["num_encoder_layers"]
        self.num_decoder_layers = options["num_decoder_layers"]
        self.dim_model = options["dim_model"]
        self.num_heads = options["num_heads"]
        self.dim_feedforward = options["dim_feedforward"]
        self.transformers_dropout = options["transformers_dropout"]

        self.lower_bound = 0
        self.upper_bound = 3

    def detect_logkey_anomaly(self, output, label):
        num_anomaly = []
        for i in range(len(label)):
            predicted = torch.argsort(output[i])[-self.num_candidates:].clone().detach().cpu().tolist()
            # print(predicted, label[i], label[i] in predicted, predicted.index(label[i]))
            if label[i] not in predicted:
                num_anomaly.append(self.num_candidates + 1)
            else:
                num_anomaly.append(predicted.index(label[i]) + 1)
        return num_anomaly

    def detect_params_anomaly(self, output, label):
        num_anomaly = 0
        for i in range(len(label)):
            error = output[i].item() - label[i].item()
            if error < self.lower_bound or error > self.upper_bound:
                num_anomaly += 1
        return num_anomaly

    def compute_anomaly(self, results, threshold=0):
        total_errors = 0
        for seq_res in results:
            # if isinstance(threshold, float):
            #     threshold = seq_res["predicted_logkey"] * threshold
            # seq_res = seq_res[1:]
            # print(max(seq_res), min(seq_res))
            error = seq_res > threshold
            total_errors += int(error)

        return total_errors

    def find_best_threshold(self, test_normal_results, test_abnormal_results, threshold_range):
        test_abnormal_length = len(test_abnormal_results)
        test_normal_length = len(test_normal_results)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        for th in threshold_range:
            FP = self.compute_anomaly(test_normal_results, th)
            TP = self.compute_anomaly(test_abnormal_results, th)
            if TP == 0:
                continue

            # Compute precision, recall and F1-measure
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            if F1 > res[-1]:
                res = [th, TP, TN, FP, FN, P, R, F1]
        return res

    def unsupervised_helper(self, model, data_iter, vocab, data_type, scale=None, min_len=0):
        normal_errors = []

        num_test = len(data_iter)
        rand_index = torch.randperm(num_test)
        rand_index = rand_index[:int(num_test * self.test_ratio)]
        data_iter = [line for line in data_iter if len(line[0]) >= min_len]
        test_results = [0] * len(data_iter)
        logs, labels = sliding_window(data_iter, vocab, window_size=self.history_size, is_train=False, data_dir=self.data_dir)
        dataset = log_dataset(logs=logs,
                              labels=labels,
                              seq=self.sequentials,
                              quan=self.quantitatives,
                              sem=self.semantics,
                              param=self.parameters)
        data_loader = DataLoader(dataset,
                                 batch_size=min(len(dataset), 4096),
                                 shuffle=True,
                                 pin_memory=False)
        tbar = tqdm(data_loader, desc="\r")
        with torch.no_grad():
            for _, (log, label) in enumerate(tbar):
                features = []
                seq_idx = log['idx'].clone().detach().cpu().tolist()
                del log['idx']
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))

                output, _ = model(features=features, device=self.device)

                if self.is_logkey:
                    num_logkey_anomaly = self.detect_logkey_anomaly(output, label)
                    for i in range(len(seq_idx)):
                        # if num_logkey_anomaly[i] > self.num_candidates:
                        #     print(seq_idx[i])
                        test_results[seq_idx[i]] = max(num_logkey_anomaly[i], test_results[seq_idx[i]])
            # print(test_results)
            return test_results, normal_errors

    def predict_unsupervised(self):
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        if self.model_name == "deeplog":
            lstm_model = deeplog
        else:
            lstm_model = loganomaly

        model_init = lstm_model(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                vocab_size=len(vocab),
                                embedding_dim=self.embedding_dim)
        model = model_init.to(self.device)

        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        test_normal_loader, test_abnormal_loader = generate(self.output_dir, 'test.pkl')
        print(len(test_normal_loader), len(test_abnormal_loader))
        # print("testing normal size: {}, testing abnormal size: {}".format(test_normal_length, test_abnormal_length))

        scale = None
        if self.is_time:
            with open(self.model_dir + "scale.pkl", "rb") as f:
                scale = pickle.load(f)

        # Test the model
        start_time = time.time()
        test_normal_results, normal_errors = self.unsupervised_helper(model, test_normal_loader, vocab, 'test_normal',
                                                                      scale=scale, min_len=self.min_len)
        test_abnormal_results, abnormal_errors = self.unsupervised_helper(model, test_abnormal_loader, vocab,
                                                                          'test_abnormal', scale=scale,
                                                                          min_len=self.min_len)

        print("Saving test normal results", self.model_dir + "test_normal_results")
        with open(self.model_dir + "test_normal_results", "wb") as f:
            pickle.dump(test_normal_results, f)

        print("Saving test abnormal results", self.model_dir + "test_abnormal_results")
        with open(self.model_dir + "test_abnormal_results", "wb") as f:
            pickle.dump(test_abnormal_results, f)

        TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(test_normal_results,
                                                                test_abnormal_results,
                                                                threshold_range=[i for i in
                                                                                 range(0, self.num_candidates + 1)])
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)
        print('Best threshold', TH)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

        if self.is_time:
            sns.kdeplot(normal_errors, label='normal errors')
            sns.kdeplot(abnormal_errors, label='abnormal errors')
            x = np.linspace(self.gaussian_mean - 3 * self.gaussian_std, self.gaussian_mean + 3 * self.gaussian_std, 100)
            plt.plot(x, stats.norm.pdf(x, self.gaussian_mean, self.gaussian_std), label='gaussian')
            plt.legend()
            print("save error distribution")
            plt.savefig(self.model_dir + 'error_distrubtion.png')
            plt.show()

    def predict_supervised(self):
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        if self.model_name == "neurallog":
            model = TransformerClassification(num_encoder_layers=self.num_encoder_layers,
                                              num_decoder_layers=self.num_decoder_layers,
                                              dim_model=self.dim_model,
                                              num_heads=self.num_heads,
                                              dim_feedforward=self.dim_feedforward,
                                              droput=self.transformers_dropout).to(self.device)
        elif self.model_name == "cnn":
            model = TextCNN(self.dim_model, self.seq_len, 128).to(self.device)
        else:
            lstm_model = robustlog

            model_init = lstm_model(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    vocab_size=len(vocab),
                                    embedding_dim=self.embedding_dim)
            model = model_init.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        test_normal_loader, test_abnormal_loader = generate(self.output_dir, 'test.pkl')
        print(len(test_normal_loader), len(test_abnormal_loader))
        start_time = time.time()
        data_iter = [line for line in test_normal_loader if len(line[0]) >= self.min_len]
        test_results = [0] * len(data_iter)
        logs, labels = sliding_window(data_iter, vocab, window_size=self.history_size, is_train=False, data_dir=self.data_dir)
        dataset = log_dataset(logs=logs,
                              labels=labels,
                              seq=self.sequentials,
                              quan=self.quantitatives,
                              sem=self.semantics,
                              param=self.parameters)
        data_loader = DataLoader(dataset,
                                 batch_size=min(len(dataset), 4096),
                                 shuffle=True,
                                 pin_memory=False)
        tbar = tqdm(data_loader, desc="\r")
        with torch.no_grad():
            for _, (log, label) in enumerate(tbar):
                features = []
                seq_idx = log['idx'].clone().detach().cpu().tolist()
                del log['idx']
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))

                output, _ = model(features=features, device=self.device)

                for i in range(len(seq_idx)):
                    test_results[seq_idx[i]] = test_results[seq_idx[i]] or \
                                               torch.argsort(output[i])[-1].clone().detach().cpu()
        FP = sum(test_results)
        TN = len(test_results) - FP

        data_iter = [line for line in test_abnormal_loader if len(line[0]) >= self.min_len]
        test_results = [0] * len(data_iter)
        logs, labels = sliding_window(data_iter, vocab, window_size=self.history_size, is_train=False,
                                      data_dir=self.data_dir)
        dataset = log_dataset(logs=logs,
                              labels=labels,
                              seq=self.sequentials,
                              quan=self.quantitatives,
                              sem=self.semantics,
                              param=self.parameters)
        data_loader = DataLoader(dataset,
                                 batch_size=min(len(dataset), 4096),
                                 shuffle=True,
                                 pin_memory=False)
        tbar = tqdm(data_loader, desc="\r")
        with torch.no_grad():
            for _, (log, label) in enumerate(tbar):
                features = []
                seq_idx = log['idx'].clone().detach().cpu().tolist()
                del log['idx']
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))

                output, _ = model(features=features, device=self.device)

                for i in range(len(seq_idx)):
                    test_results[seq_idx[i]] = test_results[seq_idx[i]] or \
                                               torch.argsort(output[i])[-1].clone().detach().cpu()
        TP = sum(test_results)
        FN = len(test_results) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        SP = TN / (TN + FP)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}, FNR: {}, FPR: {}".format(TP, TN, FP, FN, FNR, FPR))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, Specificity: {:.3f}'.format(P, R, F1, SP))

        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
