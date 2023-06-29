# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import os

from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from typing import Any
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


class Trainer:
    def __init__(self, model,
                 train_dataset,
                 valid_dataset,
                 test_dataset=None,
                 is_train=True,
                 optimizer: torch.optim.Optimizer = None,
                 no_epochs: int = 100,
                 batch_size: int = 32,
                 scheduler_type: str = 'linear',
                 warmup_rate: float = 0.1,
                 accumulation_step: int = 1,
                 decay_rate: float = 0.9,
                 logger: logging.Logger = None,
                 ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.is_train = is_train
        self.optimizer = optimizer
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.warmup_rate = warmup_rate
        self.accumulation_step = accumulation_step
        self.decay_rate = decay_rate
        self.logger = logger

    def _train_epoch(self, epoch: int, train_loader: DataLoader, device: str, scheduler: Any):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}::Train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(batch, device=device)
            loss = outputs.loss
            total_loss += loss.item()
            loss = loss / self.accumulation_step
            loss.backward()
            if (idx + 1) % self.accumulation_step == 0 or idx == len(train_loader) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()

        return total_loss / len(train_loader)

    def _valid_epoch(self, epoch: int, val_loader: DataLoader, device: str):
        self.model.eval()
        y_pred = []
        y_true = []
        losses = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch + 1}::Valid"):
                del batch['idx']
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(batch, device=device)
                loss = outputs.loss
                losses.append(loss.item())
                y_pred.append(torch.argmax(outputs.probabilities, dim=1).cpu().numpy())
                y_true.append(batch['label'].cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        loss = np.mean(losses)
        acc = accuracy_score(y_true, y_pred)
        return loss, acc

    def train(self, device: str = 'cpu', save_dir: str = None, model_name: str = None):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        num_training_steps = int(self.no_epochs * len(train_loader) / self.accumulation_step)
        num_warmup_steps = int(num_training_steps * self.warmup_rate)
        scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self.model.to(device)
        for epoch in range(self.no_epochs):
            train_loss = self._train_epoch(epoch, train_loader, device, scheduler)
            val_loss, val_acc = self._valid_epoch(epoch, val_loader, device)
            if self.logger is not None:
                self.logger.info(
                    f"Epoch {epoch + 1} || Train Loss: {train_loss} || Val Loss: {val_loss} || Val Acc: {val_acc}")
            if save_dir is not None and model_name is not None:
                self.save_model(save_dir, model_name)
        self.save_model(save_dir, model_name)
        return train_loss, val_loss, val_acc

    def predict_supervised(self, dataset, y_true, device: str = 'cpu'):
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model.eval()
        y_pred = {k: 0 for k in y_true.keys()}
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader), desc=f"Predict"):
                idxs = batch['idx'].clone().cpu().tolist()
                del batch['idx']
                batch = {k: v.to(device) for k, v in batch.items()}
                y_prob = self.model.predict(batch, device=device)
                y = torch.argmax(y_prob, dim=1).cpu().numpy().tolist()
                for idx, y_i in zip(idxs, y):
                    y_pred[idx] = y_pred[idx] | y_i

        idxs = list(y_true.keys())
        y_pred = np.array([y_pred[idx] for idx in idxs])
        y_true = np.array([y_true[idx] for idx in idxs])
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        return acc, f1, pre, rec

    def predict_unsupervised(self, dataset, y_true, topk: int, device: str = 'cpu'):
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model.eval()
        y_pred = {k: 0 for k in y_true.keys()}
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader), desc=f"Predict"):
                idxs = batch['idx'].clone().cpu().tolist()
                del batch['idx']
                batch = {k: v.to(device) for k, v in batch.items()}
                y_prob = self.model.predict(batch, device=device)
                y = torch.argsort(y_prob, dim=1, descending=True)[:, :topk].cpu().numpy().tolist()
                batch_label = batch['label'].cpu().numpy().tolist()
                for idx, y_i, label_i in zip(idxs, y, batch_label):
                    y_pred[idx] = y_pred[idx] | (label_i not in y_i)

        idxs = list(y_pred.keys())
        y_pred = np.array([y_pred[idx] for idx in idxs])
        y_true = np.array([y_true[idx] for idx in idxs])
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        return acc, f1, pre, rec

    def save_model(self, save_dir: str, model_name: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, model_name))

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))
