import datetime
import os
import re
import time

from .Instance import *
from .TensorInstances import *
from .Vocab import *

rex = ['blk_(|-)[0-9]+', '(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)']


def creat_vocab(alldatas):
    '''
    All instances for vocab generation
    :param alldatas: ALL instances list
    :return: Vocab
    '''
    tag_counter = Counter()
    for inst in alldatas:
        tag_counter[inst.tag] += 1
    return Vocab(tag_counter)


def insts_numberize(insts, vocab):
    for inst in insts:
        yield inst2id(inst, vocab)


def inst2id(inst, vocab):
    srcids = vocab.word2id(inst.src_events)
    tagid = vocab.tag2id(inst.tag)
    return srcids, tagid, inst


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        insts = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield insts


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  insts in each batch
    """

    batched_data = []
    #np.random.seed(6)
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))
    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable_with_tag_logits(batch, vocab):
    slen = len(batch[0].src_events)
    batch_size = len(batch)
    for b in range(1, batch_size):
        cur_slen = len(batch[b].src_events)
        if cur_slen > slen: slen = cur_slen

    tinst = TInstWithLogits(batch_size, slen, vocab.tag_size)

    b = 0
    for srcids, tagid, inst in insts_numberize(batch, vocab):
        tinst.src_ids.append(str(inst.src_blk))
        confidence = 0.5 * (inst.confidence)
        tinst.tags[b, tagid] = 1 - confidence
        tinst.tags[b, 1 - tagid] = confidence
        tinst.g_truth[b] = tagid
        cur_slen = len(inst.src_events)
        tinst.word_len[b] = cur_slen
        for index in range(cur_slen):
            if index >= 500:
                break
            tinst.src_words[b, index] = srcids[index]
            tinst.src_masks[b, index] = 1
        b += 1
    return tinst


def batch_variable_inst(insts, tagids, vocab, tag_logits):
    if not tag_logits:
        print('No prediction made, please check.')
        exit(-1)
    for inst, tagid, tag_logit in zip(insts, tagids, tag_logits):
        pred_tag = vocab.id2tag(tagid)
        yield Instance(inst.src_events, inst.src_blk, pred_tag, inst.type, inst.confidence,
                       tag_logit), pred_tag == inst.tag


def loadDeepLogHDFSData():
    '''
    Load HDFS data that used by DeepLog
    :return: normal and abnormal data list.
    '''
    normal = []
    abnormal = []
    with open('dataset/hdfs_train') as reader:
        for line in reader.readlines():
            line = line.strip()
            if line != '':
                normal.append([int(x) for x in line.split()])
    with open('dataset/hdfs_test_normal') as reader:
        for line in reader.readlines():
            line = line.strip()
            if line != '':
                normal.append([int(x) for x in line.split()])
    with open('dataset/hdfs_test_abnormal') as reader:
        for line in reader.readlines():
            line = line.strip()
            if line != '':
                abnormal.append([int(x) for x in line.split()])
    return normal, abnormal


def loadHDFSLogs(logID2Temp, logger):
    '''
    Load raw logs in HDFS dataset.
    :param: logID2Temp log id to template mapping, None means raw log
    :return: instances list
    '''
    instances = []
    start = time.time()

    # Refer to Drain, remove first five columns in HDFS log.
    removeCols = [0, 1, 2, 3, 4]
    logs = {}
    labels = {}
    rex = re.compile(r'blk_[-]{0,1}[0-9]+')
    filterRex = ['blk_(|-)[0-9]+', '(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', '[0-9]+',
                 '/.+', '\S+\.{1}\S+', ]
    toTemp = logID2Temp is not None
    appendedID = []
    with open('dataset/HDFS/HDFS.log', 'r', encoding='utf-8') as reader:
        id = 0
        for line in reader.readlines():
            line = line.strip()
            if line != '':
                logID = id
                id += 1
                blks = re.findall(rex, line)
                tmp = set(blks)
                if len(tmp) == 1:
                    if blks[0] not in logs.keys():
                        logs[blks[0]] = []
                        labels[blks[0]] = -1
                    if toTemp:
                        template = logID2Temp[logID]
                        logs[blks[0]].append(template)
                    else:
                        cookedLine = ' '.join(
                            [token for i, token in enumerate(line.split()) if i not in removeCols])
                        cookedLine = re.sub(r'[\-~!@#$\\.=%^&*()_+`{};\':\",\[\]<>]+', ' ', cookedLine)
                        for filtrex in filterRex:
                            cookedLine = re.sub(filtrex, '', cookedLine)
                        logs[blks[0]].append(cookedLine)
                else:
                    print('One log with %d block ids.' % (len(tmp)))
                    print(tmp)
                    exit(-1)
    logger.info('Load raw data file finished. time = %.2f' % (time.time() - start))
    with open('dataset/HDFS/anomaly_label.csv', 'r', encoding='utf-8')as reader:
        line = reader.readline()
        for line in reader.readlines():
            line = line.strip()
            if line != '':
                blk, label = line.split(',')
                if blk not in labels.keys():
                    print(blk)
                    print('Missing block id in log data file.')
                    exit(-2)
                else:
                    labels[blk] = label
    logger.info('Read log label finished. time = %.2f ' % (time.time() - start))
    for blk, seq in logs.items():
        label = labels[blk]
        events = []
        for event in seq:
            events.append('$$'.join(event.split()))
        instances.append(parseInstance(events, blk, label))
    return instances


def loadTemplates(dir, logger):
    # logTemplates.txt is the default log template file that used in Drain
    filePath = os.path.join(dir, 'logTemplates.txt')
    templates = []
    logID2Temp = {}
    if os.path.exists(filePath):
        with open(filePath, 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                if line.strip() != '':
                    templates.append('$$'.join(line.strip().split()))
                else:
                    templates.append('this_is_an_empty_event')
                pass
        logger.info('templates %d' % len(templates))

        logger.info('set templates %d ' % len(set(templates)))

        for i, template in enumerate(templates):
            fileName = 'template' + str(i + 1) + '.txt'
            filePath = os.path.join(dir, fileName)
            with open(filePath, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    line = line.strip()
                    if line != '':
                        logID2Temp[int(line)] = template
        return logID2Temp, templates
    else:
        logger.info('File not found, make sure Drain.py was run first.')


def loadBGLLogs_node_fixLength(logID2Temp, fixLength):
    '''
    Load raw logs in BGL dataset.
    :param: logID2Temp log id to template mapping, None means raw log
    :return: instances list
    '''
    print('loadBGLLogs_node_fixLength')
    check_node_list = []
    with open('dataset/BGL/BGL.log', 'r', encoding='utf-8') as raw_reader:
        # all_log = []  # every data is line
        node_log = {}
        idx = 0
        for line in raw_reader.readlines():
            tokens = line.strip().split()
            label = str(tokens[0])
            node = str(tokens[3])
            check_node_list.append(node)
            dat = str(tokens[4])
            yea = int(dat[0:4])
            mon = int(dat[5:7])
            mday = int(dat[8:10])
            hou = int(dat[11:13])
            minu = int(dat[14:16])
            seco = int(dat[17:19])
            dt = datetime.datetime(yea, mon, mday, hou, minu, seco)
            templateWord = logID2Temp[idx]
            idx += 1
            if not node in node_log.keys():
                node_log[node] = []
            node_log[node].append(BGL_Log(label, dt, templateWord))
    assert len(node_log) == len(set(check_node_list))
    print('node', len(node_log))
    print("all_log done")
    count_nor_log = 0
    count_abnor_log = 0
    for ins_list in node_log.values():
        for log_ins in ins_list:
            if log_ins.log_label == '-':
                count_nor_log += 1
            else:
                count_abnor_log += 1
    print('count_nor_log', count_nor_log)
    print('count_abnor_log', count_abnor_log)
    window_list = []

    # ------------- get instance ------------
    instId = 0
    for ins_list in node_log.values():
        window_log_events = []
        window_label = 'Normal'
        for i in range(len(ins_list)):
            if i % fixLength == 0 and i > 0:
                window_list.append(parseInstance(window_log_events, instId, window_label))
                instId += 1
                window_log_events = []
                window_label = 'Normal'
            window_log_events.append(ins_list[i].log_event)
            if not ins_list[i].log_label == '-':
                window_label = 'Anomaly'
            if i == len(ins_list) - 1:
                window_list.append(parseInstance(window_log_events, instId, window_label))
                instId += 1
    print('sequence_list', len(window_list))
    return window_list


def loadHDBscanResult(file):
    res = []
    predicts = []
    outliers = []
    instances = []
    with open(file, 'r', encoding='utf-8') as reader:
        context = []
        for line in reader.readlines():
            line = line.strip()
            if line != '':
                context.append(line)
            elif len(context) == 3:
                inst = parseHDbscanInstance(context)
                instances.append(inst)
                predicts.append(int(inst.cluster))
                outliers.append(float(inst.outlier))
                context.clear()
            else:
                context.clear()
        if len(context) == 3:
            res.append(parseHDbscanInstance(context))
            context.clear()
    return instances, predicts, outliers
