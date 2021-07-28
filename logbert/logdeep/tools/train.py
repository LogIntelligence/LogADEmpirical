#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import gc

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle

from logbert.logdeep.dataset.log import log_dataset
from logbert.logdeep.dataset.sample import sliding_window, load_features
from logbert.logdeep.tools.utils import plot_train_valid_loss
from logbert.logdeep.models.lstm import deeplog, loganomaly, robustlog
from logbert.logdeep.models.cnn import TextCNN
from logbert.neural_log.transformers import TransformerClassification


class Trainer():
    def __init__(self, options):
        self.model_name = options['model_name']
        self.model_dir = options['model_dir']
        self.data_dir = options['output_dir']
        self.vocab_path = options["vocab_path"]
        self.scale_path = options["scale_path"]
        self.emb_dir = options['data_dir']

        self.window_size = options['window_size']
        self.min_len = options["min_len"]
        self.seq_len = options["seq_len"]
        self.history_size = options['history_size']

        self.input_size = options["input_size"]
        self.hidden_size = options["hidden_size"]
        self.embedding_dim = options["embedding_dim"]
        self.num_layers = options["num_layers"]

        self.max_epoch = options['max_epoch']
        self.n_epochs_stop = options["n_epochs_stop"]
        self.device = options['device']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.batch_size = options['batch_size']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.embeddings = options['embeddings']

        self.sample = options['sample']
        self.train_ratio = options['train_ratio']
        self.train_size = options['train_size']
        self.valid_ratio = options['valid_ratio']

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]

        # transformers' parameters
        self.num_encoder_layers = options["num_encoder_layers"]
        self.num_decoder_layers = options["num_decoder_layers"]
        self.dim_model = options["dim_model"]
        self.num_heads = options["num_heads"]
        self.dim_feedforward = options["dim_feedforward"]
        self.transformers_dropout = options["transformers_dropout"]
        self.random_sample = options["random_sample"]

        # detection model: predict the next log or classify normal/abnormal
        if self.model_name == "cnn" or self.model_name == "logrobust":
            self.is_predict_logkey = False
        else:
            self.is_predict_logkey = True

        self.early_stopping = False
        self.epochs_no_improve = 0
        self.criterion = None

        print("Loading vocab")
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        if self.sample == 'sliding_window':
            print("Loading train dataset\n")
            data = load_features(self.data_dir + "train.pkl", only_normal=self.is_predict_logkey)
            n_train = int(len(data) * self.train_ratio)
            train_logs, valid_logs = data[:n_train], data[n_train:]

            train_logs, train_labels = sliding_window(train_logs,
                                                      vocab=vocab,
                                                      window_size=self.history_size,
                                                      data_dir=self.emb_dir,
                                                      is_predict_logkey=self.is_predict_logkey
                                                      )

            val_logs, val_labels = sliding_window(valid_logs,
                                                  vocab=vocab,
                                                  window_size=self.history_size,
                                                  data_dir=self.emb_dir,
                                                  is_predict_logkey=self.is_predict_logkey
                                                  )
            del data, n_train
            # del vocab
            gc.collect()
        else:
            raise NotImplementedError

        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics,
                                    param=self.parameters)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics,
                                    param=self.parameters)

        del train_logs
        del val_logs
        gc.collect()

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=False)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=False)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))

        if self.model_name == "neurallog":
            self.model = TransformerClassification(num_encoder_layers=self.num_encoder_layers,
                                                   num_decoder_layers=self.num_decoder_layers,
                                                   dim_model=self.dim_model,
                                                   num_heads=self.num_heads,
                                                   dim_feedforward=self.dim_feedforward,
                                                   droput=self.transformers_dropout).to(self.device)
        elif self.model_name == "cnn":
            print(self.dim_model, self.seq_len)
            self.model = TextCNN(self.dim_model, self.seq_len, 128).to(self.device)
        else:
            if self.model_name == "deeplog":
                lstm_model = deeplog
            elif self.model_name == "loganomaly":
                lstm_model = loganomaly
            else:
                lstm_model = robustlog

            model_init = lstm_model(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    vocab_size=len(vocab),
                                    embedding_dim=self.embedding_dim)
            self.model = model_init.to(self.device)

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        self.criterion = nn.CrossEntropyLoss()
        self.time_criterion = nn.MSELoss()

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.model_dir + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.model_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("\nStarting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()

        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        acc = 0
        total_log = 0
        for i, (log, label) in enumerate(tbar):
            features = []
            del log['idx']
            for value in log.values():
                features.append(value.clone().detach().to(self.device))
            # output is log key and timestamp
            output, _ = self.model(features=features, device=self.device)

            label = label.view(-1).to(self.device)
            loss = self.criterion(output, label)

            predicted = output.argmax(dim=1).cpu().numpy()
            label = np.array([y.cpu() for y in label])
            acc += (predicted == label).sum()
            total_log += len(label)

            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()

            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description(
                "Train loss: {0:.5f} - Train acc: {1:.2f}".format(total_losses / (i + 1), acc / total_log))

        self.log['train']['loss'].append(total_losses / num_batch)

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("\nStarting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        acc = 0
        total_log = 0
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)

        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                del log['idx']
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                output, _ = self.model(features=features, device=self.device)

                label = label.view(-1).to(self.device)
                loss = 0 if not self.is_logkey else self.criterion(output, label)

                predicted = output.argmax(dim=1).cpu().numpy()
                label = np.array([y.cpu() for y in label])
                acc += (predicted == label).sum()
                total_log += len(label)

                total_losses += float(loss)
        print("\nValidation loss:", total_losses / num_batch, "Validation accuracy:", acc / total_log)
        self.log['valid']['loss'].append(total_losses / num_batch)

        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch

            if self.is_time:
                pass

            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix=self.model_name)
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve == self.n_epochs_stop:
            self.early_stopping = True
            print("Early stopping")

    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.early_stopping:
                break
            self.train(epoch)
            self.valid(epoch)
            self.save_log()

        plot_train_valid_loss(self.model_dir)
