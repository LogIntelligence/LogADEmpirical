import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from logbert.logdeep.dataset.session import sliding_window, session_window
import shutil

# tqdm.pandas()
# pd.options.mode.chained_assignment = None  # default='warn'


# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def _count_anomaly(log_path):
    total_size = 0
    normal_size = 0
    with open(log_path, errors='ignore') as f:
        for line in f:
            total_size += 1
            if line.split('')[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))


def sample_raw_data(data_file, output_file, sample_window_size, sample_step_size):
    """
    only sample supercomputer dataset such as bgl
    """
    sample_data = []
    labels = []
    idx = 0

    with open(data_file, 'r', errors='ignore') as f:
        for line in f:
            labels.append(line.split()[0] != '-')
            sample_data.append(line)

            if len(labels) == sample_window_size:
                abnormal_rate = sum(np.array(labels)) / len(labels)
                print(f"{idx + 1} lines, abnormal rate {abnormal_rate}")
                break

            idx += 1
            if idx % sample_step_size == 0:
                print(f"Process {round(idx/sample_window_size * 100,4)} % raw data", end='\r')

    with open(output_file, "w") as f:
        f.writelines(sample_data)
    print("Sampling done")


def _file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def process_dataset(data_dir, output_dir, log_file, dataset_name, window_type, window_size, step_size, train_size):
    """
    creating log sequences by sliding window
    :param data_dir:
    :param output_dir:
    :param log_file:
    :param window_size:
    :param step_size:
    :param train_size:
    :return:
    """
    ########
    # count anomaly
    ########
    # _count_anomaly(data_dir + log_file)

    ##################
    # Transformation #
    ##################
    print("Loading", f'{output_dir}{log_file}_structured.csv')
    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')

    # build log sequences
    if window_type == "sliding":
        # data preprocess
        if 'bgl' in dataset_name:
            df["datetime"] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
        else:
            df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format='%Y-%m-%d %H:%M:%S')

        df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
        df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
        df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
        df['deltaT'].fillna(0)
        n_train = int(len(df) * train_size)
        print(n_train, len(df))
        train_window_df = sliding_window(df[["timestamp", "Label", "EventId", "deltaT"]].iloc[:n_train,:],
                                   para={"window_size": float(window_size)*60, "step_size": float(step_size) * 60})
        test_window_df = sliding_window(df[["timestamp", "Label", "EventId", "deltaT"]].iloc[n_train:,:].reset_index(drop=True),
                                   para={"window_size": float(window_size)*60, "step_size": float(step_size) * 60})

    elif window_type == "session":
        # only for hdfs
        id_regex = r'(blk_-?\d+)'
        label_dict = {}
        blk_label_file = os.path.join(data_dir, "anomaly_label.csv")
        blk_df = pd.read_csv(blk_label_file)
        for _, row in tqdm(blk_df.iterrows()):
            label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

        window_df = session_window(df, id_regex, label_dict)

    else:
        raise NotImplementedError(f"{window_type} is not implemented")

    if not os.path.exists(output_dir):
        print(f"creating {output_dir}")
        os.mkdir(output_dir)

    #########
    # Train #
    #########
    df_normal = train_window_df[train_window_df["Label"] == 0]
    # shuffle normal data
    # df_normal = df_normal.sample(frac=1).reset_index(drop=True)
    # normal_len = len(df_normal)
    # train_len = int(normal_len * train_size) if isinstance(train_size, float) else train_size

    # train = df_normal[:train_len]
    train = df_normal
    _file_generator(os.path.join(output_dir,'train'), train, ["EventId"])
    shutil.copyfile(os.path.join(output_dir, "train"), os.path.join(data_dir, "train"))

    df_abnormal = train_window_df[train_window_df["Label"] == 1]
    _file_generator(os.path.join(output_dir, 'train_abnormal'), df_abnormal, ["EventId"])
    shutil.copyfile(os.path.join(output_dir, "train_abnormal"), os.path.join(data_dir, "train_abnormal"))

    print("training size {}".format(len(train)))


    ###############
    # Test Normal #
    ###############
    # test_normal = df_normal[train_len:]
    test_normal = test_window_df[test_window_df["Label"] == 0]
    _file_generator(os.path.join(output_dir, 'test_normal'), test_normal, ["EventId"])
    shutil.copyfile(os.path.join(output_dir, 'test_normal'), os.path.join(data_dir, 'test_normal'))
    print("test normal size {}".format(len(test_normal)))


    # del df_normal
    # del train
    # del test_normal
    # gc.collect()

    #################
    # Test Abnormal #
    #################
    # df_abnormal = window_df[window_df["Label"] == 1]
    df_abnormal = test_window_df[test_window_df["Label"] == 1]
    _file_generator(os.path.join(output_dir,'test_abnormal'), df_abnormal, ["EventId"])
    shutil.copyfile(os.path.join(output_dir, 'test_abnormal'), os.path.join(data_dir, 'test_abnormal'))
    print('test abnormal size {}'.format(len(df_abnormal)))

