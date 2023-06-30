import os
import pandas as pd
from logadempirical.data.grouping import time_sliding_window, session_window, session_window_bgl, fixed_window
import pickle
from sklearn.utils import shuffle
from logging import Logger
from typing import List, Tuple


def process_dataset(logger: Logger,
                    data_dir: str,
                    output_dir: str,
                    log_file: str,
                    dataset_name: str,
                    window_type: str,
                    window_size: int,
                    step_size: int,
                    train_size: float,
                    is_chronological: bool = False,
                    session_type: str = "entry") -> Tuple[str, str]:
    """
    creating log sequences by sliding window
    :param logger:
    :param data_dir:
    :param output_dir:
    :param log_file:
    :param dataset_name:
    :param window_type:
    :param window_size:
    :param step_size:
    :param train_size:
    :param is_chronological:
    :param session_type:
    :return:
    """

    if os.path.exists(os.path.join(output_dir, "train.pkl")) and os.path.exists(os.path.join(output_dir, "test.pkl")):
        logger.info(f"Loading {output_dir}/train.pkl and {output_dir}/test.pkl")
        return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")

    logger.info(f"Loading {data_dir}/{log_file}_structured.csv")
    df = pd.read_csv(f'{data_dir}/{log_file}_structured.csv')

    # build log sequences
    if window_type == "fixed":
        df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
        n_train = int(len(df) * train_size)
        if session_type == "entry":
            sliding = fixed_window
        elif session_type == "time":
            sliding = time_sliding_window
            window_size = window_size
            step_size = step_size
        if not is_chronological:
            window_df = sliding(
                df[["Label", "EventId", "EventTemplate", "Content"]],
                window_size=window_size,
                step_size=step_size
            )
            window_df = shuffle(window_df).reset_index(drop=True)
            n_train = int(len(window_df) * train_size)
            train_window = window_df.iloc[:n_train, :].to_dict("records")
            test_window = window_df.iloc[n_train:, :].to_dict("records")
        else:
            train_window = sliding(
                df[["Label", "EventId", "EventTemplate", "Content"]].iloc[:n_train, :],
                window_size=window_size,
                step_size=step_size
            ).to_dict("records")
            test_window = sliding(
                df[["Label", "EventId", "EventTemplate", "Content"]].iloc[n_train:, :].reset_index(drop=True),
                window_size=window_size,
                step_size=step_size
            ).to_dict("records")

    elif window_type == "session":
        if dataset_name == "HDFS":
            id_regex = r'(blk_-?\d+)'
            label_dict = {}
            blk_label_file = os.path.join(data_dir, "anomaly_label.csv")
            blk_df = pd.read_csv(blk_label_file)
            for _, row in enumerate(blk_df.to_dict("records")):
                label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0
            logger.info("label dict size: {}".format(len(label_dict)))
            window_df = session_window(df, id_regex, label_dict, window_size=int(window_size))
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
        elif dataset_name == "BGL":
            window_df = session_window_bgl(df)
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
    else:
        raise NotImplementedError(f"{window_type} is not implemented")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), mode="wb") as f:
        pickle.dump(train_window, f)
    with open(os.path.join(output_dir, "test.pkl"), mode="wb") as f:
        pickle.dump(test_window, f)
    return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")


if __name__ == '__main__':
    process_dataset(Logger("BGL"),
                    data_dir="../../dataset/", output_dir="../../dataset/", log_file="BGL.log",
                    dataset_name="bgl",
                    window_type="sliding", window_size=10, step_size=10, train_size=0.8, is_chronological=True,
                    session_type="entry")