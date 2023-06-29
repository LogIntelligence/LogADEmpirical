import os
import pickle

import numpy as np
import torch
from torch.utils.data import random_split

from logadempirical.data import process_dataset
from logadempirical.data.vocab import Vocab
from logadempirical.data.feature_extraction import load_features, sliding_window
from logadempirical.data.dataset import LogDataset
from logadempirical.helpers import arg_parser, get_loggers, get_optimizer
from logadempirical.models import get_model, ModelConfig
from logadempirical.trainer import Trainer


def build_vocab(vocab_path, data_dir, train_path, embeddings, is_unsupervised=False):
    if not os.path.exists(vocab_path):
        with open(train_path, 'rb') as f:
            data = pickle.load(f)
        if is_unsupervised:
            logs = [x['EventTemplate'] for x in data if x['Label'] == 0]
        else:
            logs = [x['EventTemplate'] for x in data]
        vocab = Vocab(logs, os.path.join(data_dir, embeddings))
        logger.info(f"Vocab size: {len(vocab)}")
        logger.info(f"Save vocab in {vocab_path}")
        vocab.save_vocab(vocab_path)
    else:
        vocab = Vocab.load_vocab(vocab_path)
        logger.info(f"Vocab size: {len(vocab)}")
    return vocab


def build_model(args, vocab_size):
    criterion = torch.nn.CrossEntropyLoss()
    if args.model_name == "DeepLog":
        model_config = ModelConfig(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            criterion=criterion
        )
    elif args.model_name == "LogAnomaly":
        model_config = ModelConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            criterion=criterion,
            use_semantic=args.semantic
        )
    elif args.model_name == "LogRobust":
        model_config = ModelConfig(
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            is_bilstm=True,
            n_class=args.n_class,
            criterion=criterion
        )
    elif args.model_name == "CNN":
        model_config = ModelConfig(
            embedding_dim=args.embedding_dim,
            max_seq_len=args.history_size,
            n_class=args.n_class,
            out_channels=args.hidden_size,
            criterion=criterion
        )
    elif args.model_name == "PLELog":
        raise NotImplementedError
    else:
        raise NotImplementedError
    model = get_model(args.model_name, model_config)
    return model


def train(args, train_path, vocab, model, is_unsupervised=False):
    print("Loading train dataset\n")
    data = load_features(train_path, is_unsupervised)
    sequentials, quantitatives, semantics, labels = sliding_window(
        data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=True,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )
    dataset = LogDataset(sequentials, quantitatives, semantics, labels)
    n_valid = int(len(dataset) * args.valid_ratio)
    train_dataset, valid_dataset = random_split(dataset, [len(dataset) - n_valid, n_valid])
    logger.info(f"Train dataset: {len(train_dataset)}")
    logger.info(f"Valid dataset: {len(valid_dataset)}")
    optimizer = get_optimizer(args, model.parameters())
    trainer = Trainer(
        model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        is_train=True,
        optimizer=optimizer,
        no_epochs=args.max_epoch,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        warmup_rate=args.warmup_rate,
        accumulation_step=args.accumulation_step,
        logger=logger
    )

    trainer.train(device=args.device, save_dir=args.output_dir, model_name=args.model_name)


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    logger = get_loggers(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.window_type == "sliding":
        args.output_dir = f"{args.output_dir}/{args.dataset_name}_W{args.window_size}_S{args.step_size}_C{args.is_chronological}"
    else:
        args.output_dir = f"{args.output_dir}/{args.dataset_name}_session_W{args.window_size}_S{args.step_size}"
    train_path, test_path = process_dataset(logger, data_dir=args.data_dir, output_dir=args.output_dir,
                                            log_file=args.log_file,
                                            dataset_name=args.dataset_name, window_type=args.window_type,
                                            window_size=args.window_size, step_size=args.step_size,
                                            train_size=args.train_size, is_chronological=args.is_chronological,
                                            session_type=args.session_level)

    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)
    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"
    is_unsupervised = args.model_name in ["LogAnomaly", "DeepLog"]
    log_vocab = build_vocab(vocab_path, args.data_dir, train_path, args.embeddings, is_unsupervised=is_unsupervised)
    model = build_model(args, vocab_size=len(log_vocab))
    print(model)
    train(args, train_path, log_vocab, model, is_unsupervised=is_unsupervised)

    # options["model_dir"] = options["output_dir"] + options["model_name"] + "/"
    # options["train_vocab"] = options["output_dir"] + "train.pkl"
    # options["vocab_path"] = options["output_dir"] + options["model_name"] + "_vocab.pkl"  # pickle file
    # options["model_path"] = options["model_dir"] + options["model_name"] + ".pth"
    # options["scale_path"] = options["model_dir"] + "scale.pkl"
    #
    # os.makedirs(options["model_dir"], exist_ok=True)
    #
    # print("Save options parameters")
    # save_parameters(options, options["model_dir"] + "parameters.txt")
    #
    # # if args.model_name == "logbert":
    # #     run_logbert(options)
    # if args.model_name == "deeplog":
    #     run_deeplog(options)
    # elif args.model_name == "loganomaly":
    #     run_loganomaly(options)
    # elif args.model_name == "logrobust":
    #     run_logrobust(options)
    # elif args.model_name == "cnn":
    #     run_cnn(options)
    # elif args.model_name == "plelog":
    #     run_plelog(options)
    # elif args.model_name == "baseline":
    #     pass
    # elif args.model_name == "neurallog":
    #     run_neuralog(options)
    # else:
    #     raise NotImplementedError(f"Model {args.model_name} is not defined")

    """ To run this file, use the following command:
    python main.py
        --data_dir ../data/
        --output_dir ../output/
        --model_name deeplog
        --dataset_name bgl
        --window_type session
        --window_size 10
        --step_size 1
        --train_size 0.8
        --is_chronological True
        --session_level entry
        --log_file BGL.log
        --is_process True
        --is_instance True
        --train_file train.csv
        --test_file test.csv
        --batch_size 32
        --lr 0.01
        --num_workers 5
        --adam_beta1 0.9
        --adam_beta2 0.999
        --adam_weight_decay 0.00
        --accumulation_step 1
        --optimizer adam
        --lr_decay_ratio 0.1
        --sequential True
        --quantitative False
        --semantic False
        --sample sliding_window
        --history_size 10
        --embeddings embeddings.json
        --input_size 1
        --hidden_size 128
        --num_layers 2
        --embedding_dim 50
        --num_candidates 9
        --resume_path False
        --num_encoder_layers 1
        --dim_model 300
        --num_heads 8
        --dim_feedforward 2048
        --transformers_dropout 0.1
    """
