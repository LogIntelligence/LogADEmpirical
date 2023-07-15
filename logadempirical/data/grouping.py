import os
import re
import pandas as pd
from sklearn.utils import shuffle
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def session_window(raw_data, id_regex, label_dict, window_size=20):
    data_dict = {}  # defaultdict(list)
    raw_data = raw_data.to_dict("records")

    for idx, row in tqdm(enumerate(raw_data), total=len(raw_data)):
        blkId_list = re.findall(id_regex, row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in data_dict.keys():
                data_dict[blk_Id] = {}
                data_dict[blk_Id]['EventId'] = [row["EventId"]]
                data_dict[blk_Id]['Content'] = [row['Content']]
                data_dict[blk_Id]['EventTemplate'] = [row['EventTemplate']]
            else:
                data_dict[blk_Id]['EventId'].append(row["EventId"])
                data_dict[blk_Id]['Content'].append(row['Content'])
                data_dict[blk_Id]['EventTemplate'].append(row['EventTemplate'])

    results = []
    for k, v in data_dict.items():
        results.append({"SessionId": k, "EventId": v["EventId"], "EventTemplate": v["EventTemplate"],
                        "Content": v["Content"], "Label": label_dict[k]})
    results = shuffle(results)
    return results


def session_window_bgl(raw_data):
    data_dict = {}
    label_dict = {}
    raw_data = raw_data.to_dict("records")

    for idx, row in tqdm(enumerate(raw_data)):
        node_id = row['Node']
        label = 1 if row["Label"] != "-" else 0
        if node_id not in data_dict.keys():
            data_dict[node_id] = {}
            data_dict[node_id]['EventId'] = [row["EventId"]]
            data_dict[node_id]['Content'] = [row['Content']]
            data_dict[node_id]['EventTemplate'] = [row['EventTemplate']]
            label_dict[node_id] = [label]
        else:
            data_dict[node_id]['EventId'].append(row["EventId"])
            data_dict[node_id]['Content'].append(row['Content'])
            data_dict[node_id]['EventTemplate'].append(row['EventTemplate'])
            label_dict[node_id].append(label)

    results = []
    for k, v in data_dict.items():
        results.append({"SessionId": k, "EventId": v["EventId"], "EventTemplate": v["EventTemplate"],
                        "Content": v["Content"], "Label": label_dict[k]})
    results = shuffle(results)
    print("there are %d sessions in this dataset" % len(results))
    return results


def time_sliding_window(raw_data, window_size=60, step_size=60):
    """
    split logs into sliding windows/session
    :param raw_data: dataframe columns=[timestamp, label, eventid, time duration]
    :param window_size: seconds
    :param step_size: seconds
    :return: dataframe columns=[eventids, time durations, label]
    """
    log_size = raw_data.shape[0]
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    logkey_data, log_template_data, content_data = raw_data.iloc[:, 2], raw_data.iloc[:, 3], raw_data.iloc[:, 4]
    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + window_size
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    while end_index < log_size:
        start_time = start_time + step_size
        end_time = start_time + window_size
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

    n_sess = 0
    for (start_index, end_index) in start_end_index_pair:
        new_data.append({
            "Label": label_data[start_index:end_index].values.tolist(),
            "EventId": logkey_data[start_index: end_index].values.tolist(),
            "EventTemplate": log_template_data[start_index: end_index].values.tolist(),
            "Content": content_data[start_index: end_index].values.tolist(),
            "SessionId": n_sess
        })
        n_sess += 1

    assert len(start_end_index_pair) == len(new_data)
    # print('there are %d instances (sliding windows) in this dataset\n' % len(start_end_index_pair))
    return new_data


def fixed_window(raw_data, window_size, step_size):
    """
    split logs into sliding windows/session
    :param raw_data: dataframe columns=[timestamp, label, eventid, eventtemplate, content]
    :param window_size: number of events in a window
    :param step_size: number of events to move forward
    :return: dataframe columns=[eventids, eventtemplates, content, label]
    """
    log_size = raw_data.shape[0]
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    logkey_data, log_template_data, content_data = raw_data.iloc[:, 2], raw_data.iloc[:, 3], raw_data.iloc[:, 4]
    new_data = []
    start_end_index_pair = set()

    start_index = 0
    while start_index < log_size:
        end_index = min(start_index + window_size, log_size)
        start_end_index_pair.add(tuple([start_index, end_index]))
        start_index = start_index + step_size

    n_sess = 0
    for (start_index, end_index) in start_end_index_pair:
        new_data.append({
            "Label": label_data[start_index:end_index].values.tolist(),
            "EventId": logkey_data[start_index: end_index].values.tolist(),
            "EventTemplate": log_template_data[start_index: end_index].values.tolist(),
            "Content": content_data[start_index: end_index].values.tolist(),
            "SessionId": n_sess
        })
        n_sess += 1

    assert len(start_end_index_pair) == len(new_data)
    # print('there are %d instances (sliding windows) in this dataset\n' % len(start_end_index_pair))
    return new_data


def _custom_resampler(array_like):
    return list(array_like)


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')
