from collections import Counter
import pickle
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from typing import List, Tuple, Optional, Any, Union, Dict


def load_features(data_path, is_unsupervised=True, min_len=0):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    if is_unsupervised:
        logs = []
        for seq in data:
            if len(seq['EventTemplate']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = max(seq['Label'].tolist())
            else:
                label = seq['Label']
            if label == 0:
                logs.append((seq['EventTemplate'], label))
    else:
        logs = []
        no_abnormal = 0
        for seq in data:
            if len(seq['EventTemplate']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = seq['Label'].tolist()
                if max(label) > 0:
                    no_abnormal += 1
            else:
                label = seq['Label']
                if label > 0:
                    no_abnormal += 1
            logs.append((seq['EventTemplate'], label))
        print("Number of abnormal sessions:", no_abnormal)
    return logs


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
    log_sequences = []
    num_sessions = 0

    for idx, (templates, labels) in tqdm(enumerate(data), total=len(data),
                                         desc=f"Sliding window with size {window_size}"):
        line = list(templates)
        seq_len = max(window_size, len(line))
        line = [vocab.pad_token] * (seq_len - len(line)) + line
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
                    except KeyError:
                        pass  # ignore unseen events or padding key
                # quantitative_pattern = np.array(quantitative_pattern)[:, np.newaxis]
            sequence = {'sequential': sequential_pattern}
            if quantitative:
                sequence['quantitative'] = quantitative_pattern
            if semantic:
                sequence['semantic'] = semantic_pattern
            sequence['label'] = label
            log_sequences.append(sequence)
        num_sessions += 1

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

    logger.info(f"Number of sequences: {len(labels)}")
    logger.info(f"Number of normal sequence: {len(labels) - sum(labels)}")
    logger.info(f"Number of abnormal sequence: {sum(labels)}")

    return sequentials, quantitatives, semantics, labels
