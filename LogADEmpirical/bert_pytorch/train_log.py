from torch.utils.data import DataLoader
from logadempirical.bert_pytorch.model import BERT
from logadempirical.bert_pytorch.trainer import BERTTrainer
from logadempirical.bert_pytorch.dataset import LogDataset, WordVocab
from logadempirical.bert_pytorch.dataset.sample import generate_train_valid
from logadempirical.bert_pytorch.dataset.utils import plot_train_valid_loss

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import pandas as pd
import torch
import tqdm
from collections import defaultdict
import pickle
import os
import gc

class Trainer():
    def __init__(self, options):
        self.device = options["device"]
        self.model_dir = options["model_dir"]
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.output_path = options["output_dir"]

        self.window_size = options["window_size"]
        self.adaptive_window = options["adaptive_window"]

        self.train_ratio = options["train_ratio"]
        self.valid_ratio = options["valid_ratio"]

        self.seq_len = options["seq_len"]
        self.max_len = options["max_len"]
        self.min_len = options["min_len"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.lr = options["lr"]
        self.adam_beta1 = options["adam_beta1"]
        self.adam_beta2 = options["adam_beta2"]
        self.adam_weight_decay = options["adam_weight_decay"]

        self.log_freq = options["log_freq"]
        self.epochs = options["max_epoch"]
        self.n_epochs_stop = options["n_epochs_stop"]
        self.n_warm_up_epoch = options["n_warm_up_epoch"]

        self.deepsvdd_loss = options["deepsvdd_loss"]
        self.mask_ratio = options["mask_ratio"]
        self.hidden = options["hidden"]
        self.layers = options["layers"]
        self.attn_heads = options["attn_heads"]

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.scale = options["scale"]
        self.scale_path = options["scale_path"]

    def train(self):

        print("Loading vocab", self.vocab_path)
        vocab = WordVocab.load_vocab(self.vocab_path)
        print("vocab Size: ", len(vocab))

        print("\nLoading Train Dataset")
        logkey_train, logkey_valid, time_train, time_valid = generate_train_valid(self.output_path + "train",
                                     window_size=self.window_size,
                                     adaptive_window=self.adaptive_window,
                                     valid_size=self.valid_ratio,
                                     sample_ratio=self.train_ratio,
                                     scale=self.scale,
                                     scale_path=self.scale_path,
                                     seq_len=self.seq_len,
                                     min_len=self.min_len)

        train_dataset = LogDataset(logkey_train,time_train, vocab, seq_len=self.seq_len,
                                mask_ratio=self.mask_ratio)

        print("\nLoading valid Dataset")
        # valid_dataset = generate_train_valid(self.output_path + "train", window_size=self.window_size,
        #                              adaptive_window=self.adaptive_window,
        #                              sample_ratio=self.valid_ratio)

        valid_dataset = LogDataset(logkey_valid, time_valid, vocab, seq_len=self.seq_len,mask_ratio=self.mask_ratio)

        print("Creating Dataloader")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                      collate_fn=train_dataset.collate_fn, drop_last=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                       collate_fn=train_dataset.collate_fn, drop_last=True)
        del train_dataset
        del valid_dataset
        del logkey_train
        del logkey_valid
        del time_train
        del time_valid
        gc.collect()

        print("Building BERT model")
        bert = BERT(len(vocab), max_len=self.max_len, hidden=self.hidden, n_layers=self.layers, attn_heads=self.attn_heads,
                    is_logkey=self.is_logkey, is_time=self.is_time)

        print("Creating BERT Trainer")
        self.trainer = BERTTrainer(bert, len(vocab),
                                   train_dataloader=self.train_data_loader,
                                   valid_dataloader=self.valid_data_loader,
                                   lr=self.lr,
                                   betas=(self.adam_beta1, self.adam_beta2),
                                   weight_decay=self.adam_weight_decay,
                                   log_freq=self.log_freq,
                                   is_logkey=self.is_logkey, is_time=self.is_time,
                                   deepsvdd_loss=self.deepsvdd_loss)

        self.start_iteration(surfix_log="log")
        plot_train_valid_loss(self.model_dir)

    def start_iteration(self, surfix_log):
        print("Training Start")
        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            print("\n")
            if self.deepsvdd_loss:
                center = self.calculate_center([self.train_data_loader, self.valid_data_loader])
                self.trainer.hyper_center = center

            _, train_dist = self.trainer.train(epoch)
            avg_loss, valid_dist = self.trainer.valid(epoch)
            self.trainer.save_log(self.model_dir, surfix_log)

            if self.deepsvdd_loss:
                self.trainer.radius = self.trainer.get_radius(train_dist + valid_dist, self.trainer.nu)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.trainer.save(self.model_path)
                epochs_no_improve = 0

                if epoch > self.n_warm_up_epoch and self.deepsvdd_loss:
                    best_center = self.trainer.hyper_center
                    best_radius = self.trainer.radius
                    total_dist = train_dist + valid_dist

                    print("best radius", best_radius)
                    best_center_path = self.model_dir + "best_center.pt"
                    print("Save best center", best_center_path)
                    torch.save({"center": best_center, "radius": best_radius}, best_center_path)

                    total_dist_path = self.model_dir + "best_total_dist.pt"
                    print("save total dist: ", total_dist_path)
                    torch.save(total_dist, total_dist_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve == self.n_epochs_stop:
                print("Early stopping")
                break

    # use whole training dataset to get mean and std of time predicted error
    def get_gaussian(self, data_loader_list):
        model = torch.load(self.model_path)
        model.to(self.device)
        model.eval()

        errors = []
        error_dict = defaultdict(list)
        for data_loader in data_loader_list:
            totol_length = len(data_loader)
            data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}

                result = model.forward(data["bert_input"], data["time_input"])
                mask_time_output = result["time_output"]

                mask_index = data["time_label"] > 0
                masked_logkey = data["bert_label"][mask_index].detach().cpu().numpy()
                masked_time = data["time_label"][mask_index].detach().cpu().numpy()
                predicted_time = mask_time_output.squeeze()[mask_index].detach().cpu().numpy()

                for j in range(len(mask_index)):
                    error_dict[masked_logkey[j]].append(np.log(np.abs(predicted_time[j] - masked_time[j])))

                # error for all time difference
                error = np.log(np.abs(predicted_time - masked_time))
                errors = np.concatenate((errors, error))

        mu, sig = np.mean(errors), np.std(errors)
        # print("The Gaussian distribution of valid errors, --mean {:.4f} --std {:.4f}".format(mu, sig))

        # to calculate the mean and std of gaussian, the number of masked logkeys is greater than 3
        error_dict = {k: [np.mean(v), np.std(v)] for k, v in error_dict.items() if len(v) > 3}
        error_dict["all"] = [mu, sig]

        print(os.path.exists(self.model_dir))
        with open(self.model_dir + "error_dict.pkl", 'wb') as f:
            pickle.dump(error_dict, f)
        print("Save error dict to", self.model_dir + "error_dict.pkl")

        sns.kdeplot(errors, label="time hist")

        x = np.linspace(mu - 3 * sig, mu + 3 * sig, 100)
        plt.plot(x, norm.pdf(x, mu, sig), 'r-', label='time gaussian')

        plt.legend()
        plt.title("time_error_dist")
        plt.savefig(self.model_dir + "time_error_dist.png")
        #plt.close()
        plt.show()

    def calculate_center(self, data_loader_list):
        print("start calculate center")
        # model = torch.load(self.model_path)
        # model.to(self.device)
        with torch.no_grad():
            outputs = 0
            total_samples = 0
            for data_loader in data_loader_list:
                totol_length = len(data_loader)
                data_iter = tqdm.tqdm(enumerate(data_loader), total=totol_length)
                for i, data in data_iter:
                    data = {key: value.to(self.device) for key, value in data.items()}

                    result = self.trainer.model.forward(data["bert_input"], data["time_input"])
                    cls_output = result["cls_output"]

                    outputs += torch.sum(cls_output.detach().clone(), dim=0)
                    total_samples += cls_output.size(0)

        center = outputs / total_samples

        return center





