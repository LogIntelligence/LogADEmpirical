from tqdm import tqdm
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def generate_pairs(line, window_size):
    line = np.array(line)
    line = line[:, 0]

    seqs = []
    for i in range(0, len(line), window_size):
        seq = line[i:i + window_size]
        seqs.append(seq)
    seqs += []
    seq_pairs = []
    for i in range(1, len(seqs)):
        seq_pairs.append([seqs[i - 1], seqs[i]])
    return seqs


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    line = [ln.split(",") for ln in line.split()]

    # filter the line/session shorter than min_len
    if len(line) < min_len:
        return [], []

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze(1)
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])

    return logkey_seqs, time_seq


def generate_train_valid(data_path, window_size, adaptive_window, sample_ratio, valid_size, scale, scale_path, seq_len, min_len):
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("="*40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    session = 0
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1

        logkeys, times = fixed_window(line, window_size, adaptive_window, seq_len)
        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    if scale is not None:
        total_times = np.concatenate(np.array(time_seq_pairs), axis=0).reshape(-1,1)
        scale.fit(total_times)
        for i, tn in enumerate(time_seq_pairs):
            tn = np.array(tn).reshape(-1,1)
            time_seq_pairs[i] = scale.transform(tn).reshape(-1).tolist()

        print("Save {} in {}".format(scale, scale_path))
        with open(scale_path, "wb") as f:
            pickle.dump(scale, f)

    logkey_seq_pairs = np.array(logkey_seq_pairs)
    time_seq_pairs = np.array(time_seq_pairs)

    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(logkey_seq_pairs,
                                                                                      time_seq_pairs,
                                                                                      test_size=test_size,
                                                                                      random_state=1234)

    # sort seq_pairs by seq len
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    print("="*40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("="*40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset

