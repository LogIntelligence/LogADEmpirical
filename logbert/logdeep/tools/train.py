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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from logbert.logdeep.dataset.log import log_dataset
from logbert.logdeep.dataset.sample import sliding_window, session_window, split_features
from logbert.logdeep.tools.utils import plot_train_valid_loss
from logbert.logdeep.models.lstm import deeplog, loganomaly, robustlog


class Trainer():
    def __init__(self, options):
        self.model_name = options['model_name']
        self.model_dir = options['model_dir']
        self.data_dir = options['data_dir']
        self.vocab_path = options["vocab_path"]
        self.scale_path = options["scale_path"]

        self.window_size = options['window_size']
        self.min_len = options["min_len"]
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

        self.sample = options['sample']
        self.train_ratio = options['train_ratio']
        self.valid_ratio = options['valid_ratio']

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]

        self.early_stopping = False
        self.epochs_no_improve = 0
        self.criterion = None

        if self.sample == 'sliding_window':
            print("Loading train dataset\n")
            logkeys, times = split_features(self.data_dir + "train",
                                            self.train_ratio,
                                            scale=None,
                                            scale_path=self.scale_path,
                                            min_len=self.min_len)

            train_logkeys, valid_logkeys, train_times, valid_times = train_test_split(logkeys, times,
                                                                                      test_size=self.valid_ratio)

            print("Loading vocab")
            with open(self.vocab_path, 'rb') as f:
                vocab = pickle.load(f)

            train_logs, train_labels = sliding_window((train_logkeys, train_times),
                                                      vocab=vocab,
                                                      window_size=self.history_size,
                                                      )

            val_logs, val_labels = sliding_window((valid_logkeys, valid_times),
                                                  vocab=vocab,
                                                  window_size=self.history_size,
                                                  )
            del train_logkeys, train_times
            del valid_logkeys, valid_times
            # del vocab
            gc.collect()

        elif self.sample == 'session_window':
            (train_logs, train_labels), (val_logs, val_labels) = session_window(self.data_dir,
                                                                                datatype='train',
                                                                                sample_ratio=self.train_ratio)
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
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))

        if self.model_name == "deeplog":
            lstm_model = deeplog
        elif self.model == "loganomaly":
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

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
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
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))

            # output is log key and timestamp
            output0, output1 = self.model(features=features, device=self.device)
            output0, output1 = output0.squeeze(), output1.squeeze()

            label0, label1 = label
            label0 = label0.view(-1).to(self.device)
            label1 = label1.view(-1).to(self.device).float()

            loss0 = 0 if not self.is_logkey else self.criterion(output0, label0)
            loss1 = 0 if not self.is_time else self.time_criterion(output1, label1)
            loss = loss0 + loss1

            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()

            # Basically it involves making optimizer steps after several batches
            # thus increasing effective batch size.
            # https: // www.kaggle.com / c / understanding_cloud_organization / discussion / 105614
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))

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

        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)

        errors = []
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))

                output0, output1 = self.model(features=features, device=self.device)
                output0, output1 = output0.squeeze(), output1.squeeze()
                label0, label1 = label
                label0 = label0.view(-1).to(self.device)
                label1 = label1.view(-1).to(self.device).float()

                loss0 = 0 if not self.is_logkey else self.criterion(output0, label0)
                loss1 = 0 if not self.is_time else self.time_criterion(output1, label1)
                loss = loss0 + loss1

                if self.is_time:
                    error = output1.detach().clone().cpu().numpy() - label1.detach().clone().cpu().numpy()
                    errors = np.concatenate((errors, error))

                total_losses += float(loss)
        print("\nValidation loss:", total_losses / num_batch)
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

        # print("The Gaussian distribution of predicted errors, --mean {:.4f} --std {:.4f}".format(mean, std))
        # sns_plot = sns.kdeplot(errors)
        # sns_plot.get_figure().savefig(self.model_dir + "valid_error_dist.png")
        # plt.close()
        # print("validation error distribution saved")

    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.early_stopping:
                break
            self.train(epoch)
            self.valid(epoch)
            self.save_log()

        plot_train_valid_loss(self.model_dir)
