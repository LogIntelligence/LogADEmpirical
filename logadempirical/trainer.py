# -*- coding: utf-8 -*-
import pdb
from collections import Counter

import numpy as np
from tqdm import tqdm
import os

from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Any, Optional, List, Union, Tuple
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, top_k_accuracy_score
from itertools import chain


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
                 num_classes: int = 2,
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
        self.num_classes = num_classes
        self.scheduler = None

    def _train_epoch(self, train_loader: DataLoader, device: str, scheduler: Any, progress_bar: Any):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        # for batch in train_loader:
        #     print(batch)
        #     break
        # return 0
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

    def _valid_epoch(self, val_loader: DataLoader, device: str, topk: int = 1):
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
            probabilities = self.accelerator.gather(outputs.probabilities)
            y_pred.append(probabilities.detach().clone().cpu().numpy())
            # y_pred = self.accelerator.gather(y_pred)
            losses.append(loss.item())
            label = self.accelerator.gather(batch['label'])
            y_true.append(label.detach().clone().cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        loss = np.mean(losses)
        if topk > 1:
            for k in range(1, self.num_classes + 1):
                acc = top_k_accuracy_score(y_true, y_pred, k=k, labels=np.arange(self.num_classes))
                if acc >= 0.997:
                    self.logger.info(f"Top-{k} accuracy: {acc}")
                    return loss, acc, k
            # y_pred_indices = torch.argsort(torch.tensor(y_pred), dim=1, descending=True)[:, :]
            # pos = [np.where(y_pred_indices[i] == y_true[i])[0] for i in range(len(y_pred))]
            # # pdb.set_trace()
            # pos = [p[0] + 1 for p in pos if len(p) > 0]
            # if len(pos) > 0:
            #     # remove all 1 from pos
            #     # pos = [p for p in pos if p > 1]
            #     # self.logger.info(Counter(pos))
            #     # self.logger.info(f"Percentile: {np.percentile(pos, 0.997)} - {np.percentile(pos, 0.95)} - {np.percentile(pos, 0.001)}")
            #     p_mean = np.mean(pos)
            #     p_sd = np.std(pos)
            #     # 6 sigma rule
            #     self.logger.info(Counter(pos))
            #     pos = [p for p in pos if p_mean - 3 * p_sd <= p <= p_mean + 3 * p_sd]
            #     self.logger.info(Counter(pos))
            #     recommend_k = max(pos) if len(pos) > 0 else 1
            #     self.logger.info(f"Recommend topk: {recommend_k}")
            # else:
            #     print("No correct prediction")
            #     recommend_k = 1
            # return loss, acc, recommend_k
        else:
            acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))
            return loss, acc, 1

    def train(self, device: str = 'cpu', save_dir: str = None, model_name: str = None, topk: int = 1):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)  # ,# num_workers=5,
        # collate_fn=self.train_dataset.collate_fn)
        val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size)  # ,# num_workers=5,
        # collate_fn=self.valid_dataset.collate_fn)

        self.model.to(device)
        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )
        num_training_steps = int(self.no_epochs * len(train_loader) / self.accumulation_step)
        num_warmup_steps = int(num_training_steps * self.warmup_rate)
        self.scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps), desc=f"Training",
                            disable=not self.accelerator.is_local_main_process)
        total_train_loss = 0
        total_val_loss = 0
        total_val_acc = 0
        for epoch in range(self.no_epochs):
            train_loss = self._train_epoch(train_loader, device, self.scheduler, progress_bar)
            val_loss, val_acc, valid_k = self._valid_epoch(val_loader, device, topk=topk)
            if self.logger is not None:
                self.logger.debug(
                    f"Epoch {epoch + 1}||Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
                print(
                    f"Epoch {epoch + 1}||Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

            total_train_loss += train_loss
            total_val_loss += val_loss
            total_val_acc += val_acc
            if save_dir is not None and model_name is not None:
                self.save_model(save_dir, model_name)
        _, _, train_k = self._valid_epoch(train_loader, device, topk=topk)
        self.logger.info(f"Train top-{topk}: {train_k}, Valid top-{topk}: {valid_k}")
        self.save_model(save_dir, model_name)
        return total_train_loss / self.no_epochs, val_loss, val_acc, max(train_k, valid_k)

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
                y = self.model.predict_class(batch, device=device)
            # y = torch.argmax(y_prob, dim=1)
            y = self.accelerator.gather(y).cpu().numpy().tolist()
            for idx, y_i in zip(idxs, y):
                # print(y_i)
                # print(y_pred[idx])
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

    def predict_unsupervised(self,
                             dataset,
                             y_true, topk: int,
                             device: str = 'cpu',
                             is_valid: bool = False,
                             num_sessions: Optional[List[int]] = None
                             ) -> Union[Tuple[float, float, float, float], Tuple[float, int]]:
        def find_topk(dataloader):
            y_topk = []
            for batch in dataloader:
                # batch = {k: v.to(device) for k, v in batch.items()}
                label = self.accelerator.gather(batch['label'])
                with torch.no_grad():
                    y_prob = self.model.predict(batch, device=device)
                y_pred = torch.argsort(y_prob, dim=1, descending=True)[:, :]
                y_pos = torch.where(y_pred == label.unsqueeze(1))[1] + 1
                y_topk.extend(y_pos.cpu().numpy().tolist())
            return int(np.ceil(np.percentile(y_topk, 0.99)))

        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
        self.model.eval()
        if is_valid:
            acc, _, _, _ = self.predict_unsupervised_helper(test_loader, y_true, topk, device)
            self.logger.info(find_topk(test_loader))
            return acc, find_topk(test_loader)
        else:
            return self.predict_unsupervised_helper(test_loader, y_true, topk, device, num_sessions)

    def predict_logbert(self, valid_data_normal, valid_data_abnormal, device: str = "cpu"):
        self.model.eval()
        loss_normal = 0
        loss_abnormal = 0
        batch_normal = 0
        batch_abnormal = 0
        print("***********")
        valid_loader_normal = DataLoader(valid_data_normal, batch_size=128, collate_fn=valid_data_normal.collate_fn,
                                         num_workers=5, drop_last=True)
        valid_loader_abnormal = DataLoader(valid_data_abnormal, batch_size=128,
                                           collate_fn=valid_data_abnormal.collate_fn, num_workers=5, drop_last=True)
        for batch in valid_loader_normal:
            batch_normal += 1
            out = self.model(batch, device)
            loss_normal += out.loss
        for batch in valid_loader_abnormal:
            out = self.model(batch, device)
            loss_abnormal += out.loss
            batch_abnormal += 1
        return loss_normal / batch_normal, loss_abnormal / batch_abnormal

    def predict_unsupervised_helper(self, test_loader, y_true, topk: int, device: str = 'cpu',
                                    num_sessions: Optional[List[int]] = None) -> Tuple[float, float, float, float]:
        y_pred = {k: 0 for k in y_true.keys()}
        progress_bar = tqdm(total=len(test_loader), desc=f"Predict",
                            disable=not self.accelerator.is_local_main_process)
        for batch in test_loader:
            idxs = self.accelerator.gather(batch['idx']).detach().clone().cpu().numpy().tolist()
            support_label = (batch['sequential'] >= self.num_classes).any(dim=1)
            support_label = self.accelerator.gather(support_label).cpu().numpy().tolist()
            batch_label = self.accelerator.gather(batch['label']).cpu().numpy().tolist()
            del batch['idx']
            with torch.no_grad():
                y = self.accelerator.unwrap_model(self.model).predict_class(batch, top_k=topk, device=device)
            y = self.accelerator.gather(y).cpu().numpy().tolist()
            for idx, y_i, label_i, s_label in zip(idxs, y, batch_label, support_label):
                y_pred[idx] = y_pred[idx] | (label_i not in y_i or s_label)
            progress_bar.update(1)
        progress_bar.close()
        idxs = list(y_pred.keys())
        self.logger.info(f"Computing metrics...")
        if num_sessions is not None:
            self.logger.info(f"Total sessions: {sum(num_sessions)}")
            y_pred = [[y_pred[idx]] * num_sessions[idx] for idx in idxs]
            y_true = [[y_true[idx]] * num_sessions[idx] for idx in idxs]
            y_pred = np.array(list(chain.from_iterable(y_pred)))
            y_true = np.array(list(chain.from_iterable(y_true)))
            self.logger.info(f"Total sessions: {len(y_pred)}")
        else:
            y_pred = np.array([y_pred[idx] for idx in idxs])
            y_true = np.array([y_true[idx] for idx in idxs])
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        progress_bar.close()
        return acc, f1, pre, rec

    def save_model(self, save_dir: str, model_name: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(
            {
                "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "optimizer": self.optimizer.state_dict(),
                "model": self.model.state_dict()
            },
            f"{save_dir}/{model_name}.pt"
        )

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.accelerator.prepare(self.model)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def creat_data_loader(self):
        """
        Create dataloader by removing duplicate sequences (same sequential and label) and adding weight for each
        Returns: torch.utils.data.DataLoader
        -------
        """
        # Remove duplicate sequences
        # self.logger.info("Removing duplicate sequences")
        # self.logger.info(f"Before removing: {len(self.train_dataset)}")
        # self.logger.info(f"Before removing: {len(self.valid_dataset)}")
        # self.train_dataset.remove_duplicates()
        # self.valid_dataset.remove_duplicates()
        # self.logger.info(f"After removing: {len(self.train_dataset)}")
        # self.logger.info(f"After removing: {len(self.valid_dataset)}")

        # Add weight for each sequence
        self.logger.info("Adding weight for each sequence")
        self.logger.info(f"Before adding: {len(self.train_dataset)}")
        self.logger.info(f"Before adding: {len(self.valid_dataset)}")
        self.train_dataset.add_weight()
        self.valid_dataset.add_weight()
        self.logger.info(f"After adding: {len(self.train_dataset)}")
        self.logger.info(f"After adding: {len(self.valid_dataset)}")

        # Create dataloader
        self.logger.info("Creating dataloader")
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=self.train_dataset.collate_fn)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                                  collate_fn=self.valid_dataset.collate_fn)
        return train_loader, valid_loader