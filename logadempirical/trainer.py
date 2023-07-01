# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import os

from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from typing import Any
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class Trainer:
    def __init__(self, model,
                 train_dataset,
                 valid_dataset,
                 is_train=True,
                 optimizer: torch.optim.Optimizer = None,
                 no_epochs: int = 100,
                 batch_size: int = 32,
                 scheduler_type: str = 'linear',
                 warmup_rate: float = 0.1,
                 accumulation_step: int = 1,
                 decay_rate: float = 0.9,
                 logger: logging.Logger = None,
                 accelerator: Any = None,
                 ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.is_train = is_train
        self.optimizer = optimizer
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.warmup_rate = warmup_rate
        self.accumulation_step = accumulation_step
        self.decay_rate = decay_rate
        self.logger = logger
        self.accelerator = accelerator

    def _train_epoch(self, train_loader: DataLoader, device: str, scheduler: Any, progress_bar: Any):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(batch, device=device)
            loss = outputs.loss
            total_loss += loss.item()
            loss = loss / self.accumulation_step
            self.accelerator.backward(loss)
            if (idx + 1) % self.accumulation_step == 0 or idx == len(train_loader) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': total_loss / (idx + 1)})

        return total_loss / len(train_loader)

    def _valid_epoch(self, val_loader: DataLoader, device: str):
        self.model.eval()
        y_pred = []
        y_true = []
        losses = []
        for idx, batch in enumerate(val_loader):
            del batch['idx']
            # batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(batch, device=device)
            loss = outputs.loss
            probabilities = outputs.probabilities
            y_pred.append(torch.argmax(probabilities, dim=1).detach().clone().cpu().numpy())
            y_pred = self.accelerator.gather(y_pred)
            losses.append(loss.item())
            label = self.accelerator.gather(batch['label'])
            y_true.append(label.detach().clone().cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        loss = np.mean(losses)
        acc = accuracy_score(y_true, y_pred)
        return loss, acc

    def train(self, device: str = 'cpu', save_dir: str = None, model_name: str = None):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )
        num_training_steps = int(self.no_epochs * len(train_loader) / self.accumulation_step)
        num_warmup_steps = int(num_training_steps * self.warmup_rate)
        scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps), desc=f"Training",
                            disable=not self.accelerator.is_local_main_process)
        for epoch in range(self.no_epochs):
            train_loss = self._train_epoch(train_loader, device, scheduler, progress_bar)
            val_loss, val_acc = self._valid_epoch(val_loader, device)
            if self.logger is not None:
                self.logger.info(
                    f"Epoch {epoch + 1}::Train Loss || {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            if save_dir is not None and model_name is not None:
                self.save_model(save_dir, model_name)
        self.save_model(save_dir, model_name)
        return train_loss, val_loss, val_acc

    def predict_supervised(self, dataset, y_true, device: str = 'cpu'):
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
        self.model.eval()
        y_pred = {k: 0 for k in y_true.keys()}
        progress_bar = tqdm(total=len(test_loader), desc=f"Predict",
                            disable=not self.accelerator.is_local_main_process)
        for batch in test_loader:
            idxs = self.accelerator.gather(batch['idx']).cpu().numpy().tolist()
            del batch['idx']
            # batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                y_prob = self.model.predict(batch, device=device)
            y = torch.argmax(y_prob, dim=1)
            y = self.accelerator.gather(y).cpu().numpy().tolist()
            for idx, y_i in zip(idxs, y):
                y_pred[idx] = y_pred[idx] | y_i
            progress_bar.update(1)

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
        self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
        self.model.eval()
        y_pred = {k: 0 for k in y_true.keys()}
        progress_bar = tqdm(total=len(test_loader), desc=f"Predict",
                            disable=not self.accelerator.is_local_main_process)
        for batch in test_loader:
            idxs = self.accelerator.gather(batch['idx']).detach().clone().cpu().numpy().tolist()
            del batch['idx']
            # batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                y_prob = self.model.predict(batch, device=device)
            y = torch.argsort(y_prob, dim=1, descending=True)[:, :topk]
            y = self.accelerator.gather(y).cpu().numpy().tolist()
            batch_label = self.accelerator.gather(batch['label']).cpu().numpy().tolist()
            for idx, y_i, label_i in zip(idxs, y, batch_label):
                y_pred[idx] = y_pred[idx] | (label_i not in y_i)
            progress_bar.update(1)

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
        self.model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(self.model.state_dict(), f"{save_dir}/{model_name}.pt")

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))
