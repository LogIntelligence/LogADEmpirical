import os
import pickle
from collections import Counter

import torch
from sklearn.utils import shuffle

from logadempirical.data import process_dataset
from logadempirical.data.vocab import Vocab
from logadempirical.data.feature_extraction import load_features, sliding_window
from logadempirical.data.dataset import LogDataset
from logadempirical.helpers import arg_parser, get_optimizer
from logadempirical.models import get_model, ModelConfig
from logadempirical.trainer import Trainer
from accelerate import Accelerator
import logging
from logging import getLogger
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

accelerator = Accelerator()


def build_vocab(vocab_path, data_dir, train_path, embeddings, embedding_dim=300, is_unsupervised=False):
    """
    Build vocab from training data
    Parameters
    ----------
    vocab_path: str: Path to save vocab
    data_dir: str: Path to data directory
    train_path: str: Path to training data
    embeddings: str: Path to pretrained embeddings
    embedding_dim: int: Dimension of embeddings
    is_unsupervised: bool: Whether the model is unsupervised or not

    Returns
    -------
    vocab: Vocab: Vocabulary
    """
    if not os.path.exists(vocab_path):
        with open(train_path, 'rb') as f:
            data = pickle.load(f)
        if is_unsupervised:
            logs = [x['EventTemplate'] for x in data if np.max(x['Label']) == 0]
        else:
            logs = [x['EventTemplate'] for x in data]
        vocab = Vocab(logs, os.path.join(data_dir, embeddings), embedding_dim=embedding_dim)
        logger.info(f"Vocab size: {len(vocab)}")
        logger.info(f"Save vocab in {vocab_path}")
        vocab.save_vocab(vocab_path)
    else:
        vocab = Vocab.load_vocab(vocab_path)
        logger.info(f"Vocab size: {len(vocab)}")
    return vocab


def build_model(args, vocab_size):
    """
    Build model
    Parameters
    ----------
    args: argparse.Namespace: Arguments
    vocab_size: int: Size of vocabulary

    Returns
    -------

    """
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
            dropout=args.dropout,
            criterion=criterion
        )
    elif args.model_name == "CNN":
        model_config = ModelConfig(
            embedding_dim=args.embedding_dim,
            max_seq_len=args.history_size,
            n_class=args.n_class,
            dropout=args.dropout,
            out_channels=args.hidden_size,
            criterion=criterion
        )
    elif args.model_name == "PLELog":
        raise NotImplementedError
    else:
        raise NotImplementedError
    model = get_model(args.model_name, model_config)
    return model


def run(args, train_path, test_path, vocab, model, is_unsupervised=False):
    """
    Run model
    Parameters
    ----------
    args: argparse.Namespace: Arguments
    train_path: str: Path to training data
    test_path: str: Path to test data
    vocab: Vocab: Vocabulary
    model: torch.nn.Module: Model
    is_unsupervised: bool: Whether the model is unsupervised or not

    Returns
    -------
    Accuracy metrics
    """
    print("Loading train dataset\n")
    data, stat = load_features(train_path,
                               is_unsupervised=is_unsupervised,
                               min_len=args.history_size,
                               pad_token=vocab.pad_token,
                               is_train=True)
    logger.info(f"Train data statistics: {stat}")
    data = shuffle(data)
    n_valid = int(len(data) * args.valid_ratio)
    train_data, valid_data = data[:-n_valid], data[-n_valid:]

    sequentials, quantitatives, semantics, labels, idxs, _ = sliding_window(
        train_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=True,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )
    train_dataset = LogDataset(sequentials, quantitatives, semantics, labels, idxs)
    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        valid_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=False,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )
    valid_dataset = LogDataset(sequentials, quantitatives, semantics, labels, sequence_idxs)
    logger.info(f"Train dataset: {len(train_dataset)}")
    logger.info(f"Valid dataset: {len(valid_dataset)}")
    optimizer = get_optimizer(args, model.parameters())

    device = accelerator.device
    model = model.to(device)

    logger.info(f"Start training {args.model_name} model on {device} device")

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
        logger=logger,
        accelerator=accelerator
    )

    train_loss, val_loss, val_acc = trainer.train(device=device,
                                                  save_dir=f"{args.output_dir}/models",
                                                  model_name=args.model_name)
    if is_unsupervised:
        acc, topk_acc = trainer.predict_unsupervised(valid_dataset,
                                                     session_labels,
                                                     topk=args.topk,
                                                     device=device,
                                                     is_valid=True)
        logger.info(
            f"Validation Result:: Acc: {acc:.4f}, Top-{args.topk} Acc: {topk_acc:.4f}")
    else:
        acc, f1, pre, rec = trainer.predict_supervised(valid_dataset,
                                                       session_labels,
                                                       device=device)
        logger.info(f"Validation Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Loading test dataset\n")
    data, stat = load_features(test_path,
                               is_unsupervised=is_unsupervised,
                               min_len=args.history_size,
                               pad_token=vocab.pad_token,
                               is_train=False)
    logger.info(f"Test data statistics: {stat}")
    label_dict = {}
    counter = {}
    for (s, l) in data:
        label_dict[tuple(s)] = l
        try:
            counter[tuple(s)] += 1
        except Exception:
            counter[tuple(s)] = 1
    data = [(list(k), v) for k, v in label_dict.items()]
    num_sessions = [counter[tuple(k)] for k, _ in data]
    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=False,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )

    test_dataset = LogDataset(sequentials, quantitatives, semantics, labels, sequence_idxs)
    logger.info(f"Test dataset: {len(test_dataset)}")
    if is_unsupervised:
        acc, f1, pre, rec = trainer.predict_unsupervised(test_dataset,
                                                         session_labels,
                                                         topk=args.topk,
                                                         device=device,
                                                         is_valid=False,
                                                         num_sessions=num_sessions)
    else:
        acc, f1, pre, rec = trainer.predict_supervised(test_dataset,
                                                       session_labels,
                                                       device=device)
    logger.info(f"Test Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return acc, f1, pre, rec


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    logger = getLogger(args.model_name)
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.window_type == "sliding":
        args.output_dir = f"{args.output_dir}/{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}_train{args.train_size}"
    else:
        args.output_dir = f"{args.output_dir}/{args.dataset_name}/session/train{args.train_size}"
    train_path, test_path = process_dataset(logger, data_dir=args.data_dir, output_dir=args.output_dir,
                                            log_file=args.log_file,
                                            dataset_name=args.dataset_name, window_type=args.window_type,
                                            window_size=args.window_size, step_size=args.step_size,
                                            train_size=args.train_size, is_chronological=args.is_chronological,
                                            session_type=args.session_level)

    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)
    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"
    is_unsupervised = args.model_name in ["LogAnomaly", "DeepLog"]
    log_vocab = build_vocab(vocab_path,
                            args.data_dir,
                            train_path,
                            args.embeddings,
                            embedding_dim=args.embedding_dim,
                            is_unsupervised=is_unsupervised)
    model = build_model(args, vocab_size=len(log_vocab))
    print(model)
    run(args, train_path, test_path, log_vocab, model, is_unsupervised=is_unsupervised)

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
