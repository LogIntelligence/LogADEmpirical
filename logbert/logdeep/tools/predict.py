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

from logbert.logdeep.dataset.log import log_dataset
from logbert.logdeep.dataset.sample import session_window, sliding_window
from logbert.logdeep.models.lstm import deeplog, loganomaly


def generate(output_dir, name):
    print("Loading", output_dir + name)
    with open(output_dir + name, 'r') as f:
        data_iter = f.readlines()
    return data_iter, len(data_iter)


class Predicter():
    def __init__(self, options):
        self.output_dir = options['output_dir']
        self.model_dir = options['model_dir']
        self.vocab_path = options["vocab_path"]
        self.model_path = options['model_path']

        self.device = options['device']
        self.window_size = options['window_size']
        self.min_len = options["min_len"]

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.test_ratio = options["test_ratio"]
        self.num_candidates = options['num_candidates']

        self.input_size = options["input_size"]
        self.hidden_size = options["hidden_size"]
        self.embedding_dim = options["embedding_sim"]
        self.num_layers = options["num_layers"]

        self.batch_size = options['batch_size']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']

        self.lower_bound = 0
        self.upper_bound = 3

    def detect_logkey_anomaly(self, output, label):
        num_anomaly = 0
        for i in range(len(label)):
            predicted = torch.argsort(output[i])[-self.num_candidates:].clone().detach().cpu()
            if label[i] not in predicted:
                num_anomaly += 1
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
            if isinstance(threshold, float):
                threshold = seq_res["predicted_logkey"] * threshold

            error = (self.is_logkey and seq_res["logkey_anomaly"] > threshold) or \
                    (self.is_time and seq_res["params_anomaly"] > threshold)
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
        test_results = []
        normal_errors = []

        num_test = len(data_iter)
        rand_index = torch.randperm(num_test)
        rand_index = rand_index[:int(num_test * self.test_ratio)]

        with torch.no_grad():
            for idx, line in tqdm(enumerate(data_iter)):
                if idx not in rand_index:
                    continue

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

                if scale is not None:
                    tim = np.array(tim).reshape(-1,1)
                    tim = scale.transform(tim).reshape(-1).tolist()

                logkeys, times = [logkey.tolist()], [tim.tolist()] # add next axis

                logs, labels = sliding_window((logkeys, times), vocab, window_size=self.window_size, is_train=False)
                dataset = log_dataset(logs=logs,
                                        labels=labels,
                                        seq=self.sequentials,
                                        quan=self.quantitatives,
                                        sem=self.semantics,
                                        param=self.parameters)
                data_loader = DataLoader(dataset,
                                               batch_size=min(len(dataset), 128),
                                               shuffle=True,
                                               pin_memory=True)
                # batch_size = len(dataset)
                num_logkey_anomaly = 0
                num_params_anomaly = 0
                num_predicted_logkey = 0
                for _, (log, label) in enumerate(data_loader):
                    features = []
                    for value in log.values():
                        features.append(value.clone().detach().to(self.device))

                    output0, output1 = model(features=features, device=self.device)
                    label0, label1 = label

                    num_predicted_logkey += len(label0)

                    if self.is_logkey:
                        num_logkey_anomaly += self.detect_logkey_anomaly(output0, label0)

                    if self.is_time:
                        #normal_errors.append(output1.cpu().detach().numpy() - label)
                        num_params_anomaly += self.detect_params_anomaly(output1, label1)

                # result for line at idx
                result = {"logkey_anomaly":num_logkey_anomaly,
                          "params_anomaly": num_params_anomaly,
                          "predicted_logkey": num_predicted_logkey
                          }
                test_results.append(result)
                if idx < 10 or idx % 1000 == 0:
                    print(data_type)
                    print(result)

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

        test_normal_loader, _ = generate(self.output_dir, 'test_normal')
        test_abnormal_loader, _ = generate(self.output_dir, 'test_abnormal')
        #print("testing normal size: {}, testing abnormal size: {}".format(test_normal_length, test_abnormal_length))

        scale = None
        if self.is_time:
            with open(self.model_dir + "scale.pkl", "rb") as f:
                scale = pickle.load(f)

        # Test the model
        start_time = time.time()
        test_normal_results, normal_errors = self.unsupervised_helper(model, test_normal_loader, vocab, 'test_normal', scale=scale, min_len=self.min_len)
        test_abnormal_results, abnormal_errors = self.unsupervised_helper(model, test_abnormal_loader, vocab, 'test_abnormal', scale=scale, min_len=self.min_len)

        print("Saving test normal results", self.model_dir + "test_normal_results")
        with open(self.model_dir + "test_normal_results", "wb") as f:
            pickle.dump(test_normal_results, f)

        print("Saving test abnormal results", self.model_dir + "test_abnormal_results")
        with open(self.model_dir + "test_abnormal_results", "wb") as f:
            pickle.dump(test_abnormal_results, f)

        TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(test_normal_results,
                                                                test_abnormal_results,
                                                                threshold_range=np.arange(10))
        print('Best threshold', TH)
        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(P, R, F1))

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
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.output_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
