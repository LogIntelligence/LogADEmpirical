# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import os

from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from typing import Any
import logging


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
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}::Train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(batch, device=device)
            loss = outputs.loss
            loss = loss / self.accumulation_step
            loss.backward()
            if (idx + 1) % self.accumulation_step == 0 or idx == len(train_loader) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()

    def _valid_epoch(self, epoch: int, val_loader: DataLoader, device: str):
        self.model.eval()
        y_pred = []
        y_true = []
        losses = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch + 1}::Valid"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(batch, device=device)
                loss = outputs.loss
                losses.append(loss.item())
                y_pred.append(torch.argmax(outputs.probabilities, dim=1).cpu().numpy())
                y_true.append(batch['label'].cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        loss = np.mean(losses)
        acc = np.mean(y_pred == y_true)
        self.logger.info(f"Epoch {epoch + 1}::Valid || Loss: {loss:.4f} - Acc: {acc:.4f}")

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
            self._train_epoch(epoch, train_loader, device, scheduler)
            self._valid_epoch(epoch, val_loader, device)
            if save_dir is not None and model_name is not None:
                self.save_model(save_dir, model_name)

    def save_model(self, save_dir: str, model_name: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, model_name))
