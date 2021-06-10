
import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np


# get [log key, delta time] as input for deeplog
input_dir  = '../../data/hdfs/'
output_dir = '../output/hdfs/'  # The output directory of parsing results
#log_file   = "HDFS_2k.log.txt"  # The input log file name
log_file   = 'HDFS.log'


log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"


# mapping eventid to number
def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open (output_dir + "hdfs_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)

def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _ , row in df.items():
            if row:
                f.write("{},{:.6f} ".format(row[0], row[1]))
            for r in row[2:]:
                f.write("{},{:.6f} ".format(r[0], r[1]))
            f.write('\n')

def hdfs_sampling(log_file):
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    # preserve insertion order of items
    data_dict = defaultdict(list)
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in data_dict:
                data_dict[blk_Id] = [[row["EventId"], row['timestamp']]]
            else:
                data_dict[blk_Id].append([row["EventId"], row['timestamp']])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])


def generate_train_test(data_df, train_size=0.3):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _ , row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    data_df["Label"] = data_df["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid

    data_df["EventSequence"] = data_df["EventSequence"].apply(lambda x: cal_time_diff(x))

    normal_seq = data_df[data_df["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data
    normal_len = len(normal_seq)

    train_len = int(normal_len * train_size) if isinstance(train_size, float) else train_size
    train = normal_seq.iloc[:train_len]
    df_to_file(train, output_dir + "train")
    print("training size {}".format(train_len))

    test_normal = normal_seq.iloc[train_len:]
    df_to_file(test_normal, output_dir + "test_normal")
    print("test normal size {}".format(normal_len - train_len))

    test_abnormal = data_df[data_df["Label"] == 1]["EventSequence"]
    df_to_file(test_abnormal, output_dir + "test_abnormal")
    print('test abnormal size {}'.format(len(test_abnormal)))



if __name__ == "__main__":
    #mapping()
    #hdfs_sampling(log_structured_file)
    generate_train_test(log_sequence_file, ratio=0.5)

