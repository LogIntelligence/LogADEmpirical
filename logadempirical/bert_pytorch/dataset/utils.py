import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))

# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def plot_train_valid_loss(save_dir):
    train_loss = pd.read_csv(save_dir + "train_log.csv")
    valid_loss = pd.read_csv(save_dir + "valid_log.csv")
    sns.lineplot(x="epoch",y="loss" , data = train_loss, label="train loss")
    sns.lineplot(x="epoch",y="loss" , data = valid_loss, label="valid loss")
    plt.title("epoch vs train loss vs valid loss")
    plt.legend()
    plt.savefig(save_dir+"train_valid_loss.png")
    plt.show()
    print("plot done")