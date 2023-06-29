import os
from argparse import ArgumentParser
import logging
from torch import optim


def arg_parser():
    """
    add parser parameters
    :return:
    """
    parser = ArgumentParser()
    # input parameters
    parser.add_argument("--model_name", help="which model to use", default="DeepLog",
                        choices=["DeepLog", "LogAnomaly", "LogRobust", "NeuralLog", "CNN", "PLELog"])
    parser.add_argument("--dataset_name", help="which dataset to use", default="HDFS",
                        choices=["HDFS", "BGL", "Thunderbird", "Spirit", "Hadoop"])
    parser.add_argument("--device", help="hardware device", default="cpu")
    parser.add_argument("--data_dir", default="./dataset/", metavar="DIR", help="data directory")
    parser.add_argument("--output_dir", default="./output", metavar="DIR", help="output directory")
    parser.add_argument("--log_file", default="HDFS.log", help="log file name")

    # experimental settings parameters
    parser.add_argument("--is_chronological", default=False, action='store_true', help="if use chronological split")
    parser.add_argument("--n_class", default=2, type=int, help="number of classes")

    # data process parameters
    parser.add_argument("--window_type", type=str, choices=["sliding", "session"],
                        help="window for building log sequence")
    parser.add_argument("--session_level", type=str, choices=["entry", "minute"],
                        help="to use log entries or log minutes for session level window")
    parser.add_argument('--window_size', default=5, type=int, help='window size (entries or minutes)')
    parser.add_argument('--step_size', default=1, type=int, help='step size (entries or minutes)')
    parser.add_argument('--train_size', default=0.4, type=float, help="train size")
    parser.add_argument("--valid_ratio", default=0.1, type=float, help="valid size")

    # training parameters
    parser.add_argument("--max_epoch", default=200, type=int, help="max number of training epochs")
    parser.add_argument("--warmup_rate", default=0.1, type=float, help="warmup rate for learning rate scheduler")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--scheduler", default="linear", help="learning rate scheduler type",
                        choices=["linear", "cosine", "polynomial", "constant"])

    parser.add_argument("--accumulation_step", default=1, type=int, help="gradient accumulation step")
    parser.add_argument("--optimizer", default="adam", help="optimizer type",
                        choices=["adam", "sgd", "adamw", "adagrad", "adadelta", "rmsprop"])
    parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam beta1")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam beta2")
    parser.add_argument("--epsilon", default=1e-8, type=float, help="epsilon")
    parser.add_argument("--optim_momentum", default=0.9, type=float, help="momentum for optimizer")

    # feature parameters
    parser.add_argument("--sequential", default=False, help="to use logkey sequence", action='store_true')
    parser.add_argument("--quantitative", default=False, help="to use count vector", action='store_true')
    parser.add_argument("--semantic", default=False, action='store_true',
                        help="to use semantic vector (word2vec)")

    # model parameters (deeplog, loganomaly, logrobust)
    parser.add_argument("--history_size", default=10, type=int, help="window size for entry-level detection")
    parser.add_argument("--embeddings", default="embeddings.json", help="template embedding json file")

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


def get_optimizer(args, model_parameters):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model_parameters,
                               lr=args.lr,
                               betas=(args.adam_beta1, args.adam_beta2),
                               eps=args.epsilon,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model_parameters,
                              lr=args.lr,
                              momentum=args.optim_momentum,
                              weight_decay=args.weight_decays)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model_parameters,
                                lr=args.lr,
                                betas=(args.adam_beta1, args.adam_beta2),
                                eps=args.epsilon,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model_parameters,
                                  lr=args.lr,
                                  eps=args.epsilon,
                                  weight_decay=args.weight_decay)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model_parameters,
                                   lr=args.lr,
                                   eps=args.epsilon,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model_parameters,
                                  lr=args.lr,
                                  momentum=args.optim_momentum,
                                  eps=args.epsilon,
                                  weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer