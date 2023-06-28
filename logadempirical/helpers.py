import os
from argparse import ArgumentParser
import logging


def arg_parser():
    """
    add parser parameters
    :return:
    """
    parser = ArgumentParser()
    # input parameters
    parser.add_argument("--model_name", help="which model to train", choices=["DeepLog", "LogAnomaly", "LogRobust",
                                                                              "NeuralLog", "CNN", "PLELog"])
    parser.add_argument("--dataset_name", help="which dataset to use",
                        choices=["HDFS", "BGL", "Thunderbird", "Spirit", "Hadoop"])
    parser.add_argument("--device", help="hardware device", default="cpu")
    parser.add_argument("--data_dir", default="./dataset/", metavar="DIR", help="data directory")
    parser.add_argument("--output_dir", default="./experimental_results/RQ1/random/", metavar="DIR",
                        help="output directory")
    parser.add_argument("--log_file", default="HDFS.log", help="log file name")

    # experimental settings parameters
    parser.add_argument("--is_chronological", default=False, action='store_true', help="if use chronological split")
    parser.add_argument("--n_class", default=2, type=int, help="number of classes")

    # data process parameters
    parser.add_argument("--train_file", default="train_fixed100_instances.pkl", help="train instances file name")
    parser.add_argument("--test_file", default="test_fixed100_instances.pkl", help="test instances file name")
    parser.add_argument("--window_type", type=str, choices=["sliding", "session"],
                        help="window for building log sequence")
    parser.add_argument("--session_level", type=str, choices=["entry", "minute"],
                        help="window for building log sequence")
    parser.add_argument('--window_size', default=5, type=int, help='window size (entries)')
    parser.add_argument('--step_size', default=1, type=int, help='step size (entries)')
    parser.add_argument('--train_size', default=0.4, type=float, help="train size", metavar="float or int")
    parser.add_argument("--valid_ratio", default=0.1, type=float)

    # training parameters
    parser.add_argument("--max_epoch", default=200, type=int, help="epochs")
    parser.add_argument("--n_epochs_stop", default=10, type=int,
                        help="training stops after n epochs without improvement")
    parser.add_argument("--warmup_rate", default=0.1, type=float, help="warmup rate for learning rate scheduler")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--scheduler", default="linear", help="learning rate scheduler",
                        choices=["linear", "cosine", "polynomial", "constant"])

    parser.add_argument("--accumulation_step", default=1, type=int, help="let optimizer steps after several batches")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-8, type=float)

    # feature parameters
    parser.add_argument("--sequential", default=False, help="sequences of logkeys", action='store_true')
    parser.add_argument("--quantitative", default=False, help="logkey count vector", action='store_true')
    parser.add_argument("--semantic", default=False, action='store_true',
                        help="logkey embedding with semantics vectors")

    # model parameters (deeplog, loganomaly, logrobust)
    parser.add_argument("--sample", default="sliding_window", help="split sequences by sliding window")
    parser.add_argument("--history_size", default=10, type=int, help="window size for deeplog and log anomaly")
    parser.add_argument("--embeddings", default="embeddings.json", help="template embedding json file")

    parser.add_argument("--input_size", default=1, type=int, help="input size in lstm")
    parser.add_argument("--hidden_size", default=128, type=int, help="hidden size in lstm")
    parser.add_argument("--num_layers", default=2, type=int, help="num of lstm layers")
    parser.add_argument("--embedding_dim", default=50, type=int, help="embedding dimension of logkeys")
    parser.add_argument("--num_candidates", default=9, type=int, help="top g candidates are normal")
    parser.add_argument("--resume_path", action='store_true')

    # neural_log
    parser.add_argument("--num_encoder_layers", default=1, type=int, help="number of encoder layers")
    parser.add_argument("--dim_model", default=300, type=int, help="model's dim")
    parser.add_argument("--num_heads", default=8, type=int, help="number of attention heads")
    parser.add_argument("--dim_feedforward", default=2048, type=int, help="feed-forward network's dim")

    # common
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout rate of transformers model")

    return parser


def get_loggers(model_name):
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
