from collections import Counter
import pickle
from tqdm import tqdm
from typing import List, Tuple, Optional, Any
import numpy as np


def load_features(data_path, is_unsupervised=True, min_len=0, pad_token='padding', is_train=True):
    """
    Load features from pickle file
    Parameters
    ----------
    data_path: str: Path to pickle file
    is_unsupervised: bool: Whether the model is unsupervised or not
    min_len: int: Minimum length of log sequence
    pad_token: str: Padding token
    is_train: bool: Whether the data is training data or not
    Returns
    -------
    logs: List[Tuple[List[str], int]]: List of log sequences
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    if is_train:
        if is_unsupervised:
            logs = []
            no_abnormal = 0
            for seq in data:
                seq['EventTemplate'] = seq['EventTemplate'] + [pad_token] * (min_len - len(seq['EventTemplate']) + 1)
                if not isinstance(seq['Label'], int):
                    label = max(seq['Label'])
                else:
                    label = seq['Label']
                if label == 0:
                    logs.append((seq['EventTemplate'], label))
                else:
                    no_abnormal += 1
            print("Number of abnormal sessions:", no_abnormal)
        else:
            logs = []
            no_abnormal = 0
            for seq in data:
                if len(seq['EventTemplate']) < min_len:
                    seq['EventTemplate'] = seq['EventTemplate'] + [pad_token] * (min_len - len(seq['EventTemplate']))
                if not isinstance(seq['Label'], int):
                    label = seq['Label']
                    if max(label) > 0:
                        no_abnormal += 1
                else:
                    label = seq['Label']
                    if label > 0:
                        no_abnormal += 1
                logs.append((seq['EventTemplate'], label))
            print("Number of abnormal sessions:", no_abnormal)
    else:
        logs = []
        no_abnormal = 0
        for seq in data:
            seq['EventTemplate'] = seq['EventTemplate'] + [pad_token] * (
                    min_len - len(seq['EventTemplate']) + is_unsupervised)
            if not isinstance(seq['Label'], int):
                label = seq['Label']
                if max(label) > 0:
                    no_abnormal += 1
            else:
                label = seq['Label']
                if label > 0:
                    no_abnormal += 1
            logs.append((seq['EventTemplate'], label))
        print("Number of abnormal sessions:", no_abnormal)
    logs_len = [len(log[0]) for log in logs]
    return logs, {"min": min(logs_len), "max": max(logs_len), "mean": np.mean(logs_len)}


def sliding_window(data: List[Tuple[List[str], int]],
                   window_size: int,
                   is_train: bool = True,
                   vocab: Optional[Any] = None,
                   is_unsupervised: bool = True,
                   sequential: bool = False,
                   quantitative: bool = False,
                   semantic: bool = False,
                   logger: Optional[Any] = None,
                   ) -> Any:
    """
    Sliding window for log sequence
    Parameters
    ----------
    data: List[Tuple[List[str], int]]: List of log sequences
    window_size: int: Size of sliding window
    is_train: bool: training mode or not
    vocab: Optional[Any]: Vocabulary
    is_unsupervised: bool: Whether the model is unsupervised or not
    sequential: bool: Whether to use sequential features
    quantitative: bool: Whether to use quantitative features
    semantic: bool: Whether to use semantic features
    logger: Optional[Any]: Logger

    Returns
    -------
    lists of sequential, quantitative, semantic features, and labels
    """
    log_sequences = []
    session_labels = {}
    unique_ab_events = set()

    for idx, (templates, labels) in tqdm(enumerate(data), total=len(data),
                                         desc=f"Sliding window with size {window_size}"):
        line = list(templates)
        # seq_len = max(window_size, len(line))
        # line = [vocab.pad_token] * (seq_len - len(line)) + line
        session_labels[idx] = labels if isinstance(labels, int) else max(labels)
        for i in range(len(line) - window_size if is_unsupervised else len(line) - window_size + 1):
            if is_unsupervised:
                label = vocab.get_event(line[i + window_size])
            else:
                if not isinstance(labels, int):
                    label = max(labels[i: i + window_size])
                else:
                    label = labels
            seq = line[i: i + window_size]
            sequential_pattern = [vocab.get_event(event, use_similar=quantitative) for event in seq]
            semantic_pattern = None
            if semantic:
                semantic_pattern = [vocab.get_embedding(event) for event in seq]
            quantitative_pattern = None
            if quantitative:
                quantitative_pattern = [0] * len(vocab)
                log_counter = Counter(sequential_pattern)
                for key in log_counter:
                    try:
                        quantitative_pattern[key] = log_counter[key]
                    except Exception as _:
                        pass  # ignore unseen events or padding key

            sequence = {'sequential': sequential_pattern}
            if quantitative:
                sequence['quantitative'] = quantitative_pattern
            if semantic:
                sequence['semantic'] = semantic_pattern
            sequence['label'] = label
            sequence['idx'] = idx
            log_sequences.append(sequence)

    if is_train and not is_unsupervised:
        normal_dict = {hash(tuple(seq['sequential'])): 0 for seq in log_sequences if seq['label'] == 0}
        for i, seq in enumerate(log_sequences):
            try:
                log_sequences[i]['label'] = normal_dict[hash(tuple(seq['sequential']))]
            except KeyError:
                pass

    sequentials, quantitatives, semantics = None, None, None
    if sequential:
        sequentials = [seq['sequential'] for seq in log_sequences]
    if quantitative:
        quantitatives = [seq['quantitative'] for seq in log_sequences]
    if semantic:
        semantics = [seq['semantic'] for seq in log_sequences]
    labels = [seq['label'] for seq in log_sequences]
    sequence_idxs = [seq['idx'] for seq in log_sequences]
    logger.info(f"Number of sequences: {len(labels)}")
    if not is_unsupervised:
        logger.info(f"Number of normal sequence: {len(labels) - sum(labels)}")
        logger.info(f"Number of abnormal sequence: {sum(labels)}")

    logger.warning(f"Number of unique abnormal events: {len(unique_ab_events)}")
    logger.info(f"Number of abnormal sessions: {sum(session_labels.values())}/{len(session_labels)}")

    return sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels
