#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import gc

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest as iForest

from logadempirical.logdeep.dataset.log import log_dataset
from logadempirical.logdeep.dataset.sample import sliding_window, load_features
from logadempirical.logdeep.tools.utils import plot_train_valid_loss
from logadempirical.logdeep.models.lstm import deeplog, loganomaly, robustlog
from logadempirical.logdeep.models.autoencoder import AutoEncoder
from logadempirical.logdeep.models.cnn import TextCNN
from logadempirical.neural_log.transformers import NeuralLog


class Trainer():
    def __init__(self, options):
        self.model_name = options['model_name']
        self.model_dir = options['model_dir']
        self.model_path = options['model_path']
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
        if self.model_name in ["cnn", "logrobust", "autoencoder", "neurallog"]:
            self.is_predict_logkey = False
        else:
            self.is_predict_logkey = True

        self.early_stopping = False
        self.epochs_no_improve = 0
        self.criterion = None

        print("Loading vocab")
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(len(vocab))

        if self.sample == 'sliding_window':
            print("Loading train dataset\n")
            data = load_features(self.data_dir + "train.pkl", only_normal=self.is_predict_logkey)
            n_train = int(len(data) * self.train_size)
            train_logs, train_labels = sliding_window(data,
                                                      vocab=vocab,
                                                      window_size=self.history_size,
                                                      data_dir=self.emb_dir,
                                                      is_predict_logkey=self.is_predict_logkey,
                                                      semantics=self.semantics,
                                                      sample_ratio=self.train_ratio,
                                                      e_name=self.embeddings,
                                                      in_size=self.input_size
                                                      )

            train_logs, train_labels = shuffle(train_logs, train_labels)
            # train_logs = train_logs[:200000]
            # train_labels = train_labels[:200000]
            n_val = int(len(train_logs) * self.valid_ratio)
            val_logs, val_labels = train_logs[-n_val:], train_labels[-n_val:]
            del data
            gc.collect()
        else:
            raise NotImplementedError

        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels)

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

        self.threshold_rate = self.num_train_log // self.num_valid_log

        if self.model_name == "cnn":
            print(self.dim_model, self.seq_len)
            self.model = TextCNN(self.dim_model, self.seq_len, 128).to(self.device)
        elif self.model_name == "autoencoder":
            self.model = AutoEncoder(self.hidden_size, self.num_layers, embedding_dim=self.embedding_dim).to(
                self.device)
        elif self.model_name == "neurallog":
            self.model = NeuralLog(num_encoder_layers=1, num_heads=12, dim_model=768, dim_feedforward=2048,
                                   droput=0.2).to(self.device)
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
                betas=(0.9, 0.999)
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
            del log['idx']
            features = [x.to(self.device) for x in log['features']]
            output, _ = self.model(features=features, device=self.device)
            if isinstance(output, dict):
                loss = output['loss']
                total_log += len(label)

                total_losses += float(loss)
                loss /= self.accumulation_step
                loss.backward()
                if (i + 1) % self.accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                tbar.set_description(
                    "Train loss: {0:.8f}".format(total_losses / (i + 1)))
            else:
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
                    "Train loss: {0:.8f} - Train acc: {1:.2f}".format(total_losses / (i + 1), acc / total_log))

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
                del log['idx']
                features = [x.to(self.device) for x in log['features']]
                output, _ = self.model(features=features, device=self.device)
                if isinstance(output, dict):
                    loss = output['loss']
                else:
                    label = label.view(-1).to(self.device)
                    loss = 0 if not self.is_logkey else self.criterion(output, label)

                    predicted = torch.max(output.softmax(dim=-1), 1).indices.cpu().numpy()
                    label = np.array([y.cpu() for y in label])
                    acc += (predicted == label).sum()
                    total_log += len(label)

                total_losses += float(loss)
        if total_log:
            print("\nValidation loss:", total_losses / num_batch, "Validation accuracy:", acc / total_log)
        else:
            print("\nValidation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch
            self.epochs_no_improve = 0
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix=self.model_name)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve == self.n_epochs_stop:
            self.early_stopping = True
            print("Early stopping")
        return total_losses / num_batch

    def train_autoencoder2(self):
        print("Compute representation of log sequences...")
        logs = []
        tbar = tqdm(self.valid_loader, desc="\r")
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        self.model.eval()
        with torch.no_grad():
            for i, (log, label) in enumerate(tbar):
                embs = log['features'][2].numpy()
                del log['idx']
                features = [x.to(self.device) for x in log['features']]
                output, _ = self.model(features=features, device=self.device)
                repr = output['repr'].clone().detach().cpu().numpy()
                for j in range(len(repr)):
                    logs.append((embs[j], repr[j]))
        print(logs[0][0].shape)
        reprs = np.array([log[1] for i, log in enumerate(logs)])
        print("Find normal logs...")
        iforest = iForest(n_estimators=100, max_samples="auto", contamination="auto", verbose=1)
        iforest.fit(reprs)
        y_pred = iforest.predict(reprs)
        y_pred = np.where(y_pred > 0, 0, 1)
        normal_logs = [log[0] for i, log in enumerate(logs) if y_pred[i] == 0]
        print(len(normal_logs))

        class AEDataset(Dataset):
            def __init__(self, logs):
                self.logs = logs

            def __len__(self):
                return len(self.logs)

            def __getitem__(self, idx):
                return self.logs[idx]

        dataset = AEDataset(normal_logs)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            pin_memory=False)

        print("Train second autoencoder...")
        model_ae2 = AutoEncoder(self.hidden_size, self.num_layers, embedding_dim=self.embedding_dim).to(self.device)
        model_ae2.train()
        optimizer = torch.optim.Adam(
            model_ae2.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
        )

        total_losses = 0
        optimizer.zero_grad()
        for epoch in range(0, 20):
            print("Epoch {}...".format(epoch + 1))
            tbar = tqdm(loader, desc="\r")
            for i, log in enumerate(tbar):
                features = log.to(self.device)
                output, _ = model_ae2(features=[0, 0, features], device=self.device)
                loss = output['loss']
                total_losses += float(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tbar.set_description(
                    "Train loss: {0:.5f}".format(total_losses / (i + 1)))
        recst_value = []
        # model_ae2 = best_model
        model_ae2.eval()
        print("Compute threshold...")
        for i, log in enumerate(tbar):
            features = log.to(self.device)
            output, _ = model_ae2(features=[0, 0, features], device=self.device)
            y_pred = output['y_pred']
            recst_value.extend(y_pred.clone().detach().cpu().numpy().tolist())
        from statistics import stdev
        return model_ae2, stdev(recst_value)  # * self.threshold_rate

    def start_train(self):
        val_loss = 0
        n_epoch = 0
        n_val_epoch = 0
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.early_stopping:
                break
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in (50, 75):
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.train(epoch)
            n_epoch += 1
            if epoch > 0:
                val_loss += self.valid(epoch)
                # self.save_checkpoint(epoch,
                #                      save_optimizer=False,
                #                      suffix=self.model_name)
                n_val_epoch += 1
            self.save_log()
        plot_train_valid_loss(self.model_dir)
        if self.model_name == "autoencoder":
            return self.train_autoencoder2()  # self.model, val_loss / n_val_epoch
