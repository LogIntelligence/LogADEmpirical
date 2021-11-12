import math

from scipy.spatial.distance import cdist

from logadempirical.PLELog.data.DataLoader import *

#random.seed(6)
#np.random.seed(6)

idx = 0
dup = {}
dup["Normal"] = set()
dup["Anomaly"] = set()
def process(instance, label):
    global idx
    global dup
    events = instance.src_events
    if len(events) < 10:
        seq = " ".join(events)
        if seq in dup[label]:
            return []
        dup[label].add(seq)
        idx += 1
        return [parseInstance(events, idx - 1, label)]
    res = []
    for i in range(len(events) - 10 + 1):
        seq = " ".join(events[i: i + 10].copy())
        if seq in dup[label]:
            continue
        dup[label].add(seq)
        res.append(parseInstance(events[i: i + 10].copy(), idx, label))
        idx += 1
    return res

def option_0(predicts, instances, labelledNum, logger):
    results = {}
    normals = []
    abnormals = []
    normalCores = {}
    trainReprs = []
    for instance in instances:
        trainReprs.append(instance.repr)
    TP, TN, FP, FN = 0, 0, 0, 0
    num_of_neg1 = 0
    # Read predict results and get normal cores
    for i, res in enumerate(predicts):
        if res not in results.keys():
            results[res] = []
        results[res].append(i)
        if i < labelledNum and instances[i].type == 'Normal':
            normals.extend(process(instances[i], "Normal"))
            if res not in normalCores.keys():
                normalCores[res] = 0
            normalCores[res] += 1

    for res, ids in results.items():
        if res == -1:
            label_data = []
            label_id = []
            unlabel_id = []
            unlabel2mindist = {}
            unlabel2label_dist_0_id = []

            for id in ids:
                if id < labelledNum:
                    label_data.append(instances[id].repr)
                    label_id.append(id)
                else:
                    unlabel_id.append(id)

            logger.info('-1 label nor:' + str(len(label_id)))
            logger.info('-1 unlabel:' + str(len(unlabel_id)))
            num_of_neg1 = len(label_id) + len(unlabel_id)
            # -1类的
            TN_1 = 0
            FN_1 = 0
            FP_1 = 0
            TP_1 = 0

            for id in unlabel_id:
                dists = cdist([trainReprs[id]], label_data)
                if dists.min() == 0:
                    unlabel2label_dist_0_id.append(id)
                else:
                    unlabel2mindist[id] = dists.min()

            for id in unlabel2label_dist_0_id:
                normals.extend(process(instances[id], "Normal"))#append(parseInstance(instances[id].src_events, id, 'Normal'))
                if instances[id].type == 'Normal':
                    TN += 1
                    TN_1 += 1
                else:
                    FN += 1
                    FN_1 += 1

            for id, dist in unlabel2mindist.items():
                abnormals.extend(process(instances[id], "Anomaly"))#append(parseInstance(instances[id].src_events, id, 'Anomaly'))
                if instances[id].type == 'Normal':
                    FP += 1
                    FP_1 += 1
                else:
                    TP += 1
                    TP_1 += 1


        elif res not in normalCores:
            for id in ids:
                abnormals.extend(process(instances[id], "Anomaly"))#append(parseInstance(instances[id].src_events, id, 'Anomaly'))
                if instances[id].type == 'Normal':
                    FP += 1
                else:
                    TP += 1

        # Use a threshold to decide whether those instances classified in normal cores are anomaly or not.
        else:
            for id in ids:
                if id >= labelledNum:
                    normals.extend(process(instances[id], "Normal"))#append(parseInstance(instances[id].src_events, id, 'Normal'))
                    if instances[id].type == 'Normal':
                        TN += 1
                    else:
                        FN += 1
    if TP + FP != 0 and TP + FN != 0:
        precision = 100 * TP / (TP + FP)
        recall = 100 * TP / (TP + FN)
        f = 2 * precision * recall / (precision + recall)
        logger.info('DBSCAN: TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
        logger.info('Classify finished, precision = %.2f, recall = %.2f, f = %.2f'
                    % (precision, recall, f))
    else:
        precision, recall, f = 0, 0, 0
    abnormals = [x for x in abnormals if " ".join(x.src_events) not in dup['Normal']]
    normals.extend(abnormals)
    return normals, precision, recall, f, num_of_neg1


def option_1(predicts, outlier_scores, instances, labelledNum, logger):
    global dup
    num_outlier_0 = 0
    results = {}
    normals = []
    abnormals = []
    normalCores = {}
    TP, TN, FP, FN = 0, 0, 0, 0
    normalOutlierCounter = Counter()
    anomalyOutlierCounter = Counter()
    # Read predict results and get normal cores
    print(len(predicts), len(instances))
    for i, res in enumerate(predicts):
        if res not in results.keys():
            results[res] = []
        results[res].append(i)
        if instances[i].type == 'Normal' and i < labelledNum:
            normals.extend(process(instances[i], "Normal"))
            num_outlier_0 += 1
            normalOutlierCounter[10] += 1
            if res not in normalCores.keys():
                normalCores[res] = 0
            normalCores[res] += 1

    logger.info('There are total %d clusters after hdbscan.' % len(results))
    num_of_neg1 = 0
    for res, ids in results.items():
        if res == -1:
            logger.info("cluster -1 has %d instances." % len(ids))
            label_data = []
            label_id = []
            unlabel_id = []
            unlabel2mindist = {}
            unlabel2label_dist_0_id = []

            for id in ids:
                if id < labelledNum:
                    label_data.append(instances[id].repr)
                    label_id.append(id)
                else:
                    unlabel_id.append(id)

            logger.info('-1 label nor:' + str(len(label_id)))
            logger.info('-1 unlabel:' + str(len(unlabel_id)))
            num_of_neg1 = len(label_id) + len(unlabel_id)

            for id in unlabel_id:
                dists = cdist([instances[id].repr], label_data)
                if dists.min() == 0:
                    unlabel2label_dist_0_id.append(id)
                else:
                    unlabel2mindist[id] = dists.min()

            # -1类的
            TN_1 = 0
            FN_1 = 0
            FP_1 = 0
            TP_1 = 0
            for id in unlabel2label_dist_0_id:
                normals.extend(process(instances[id], "Normal"))
                num_outlier_0 += 1
                normalOutlierCounter[10] += 1
                if instances[id].type == 'Normal':
                    TN += 1
                    TN_1 += 1
                else:
                    FN += 1
                    FN_1 += 1

            for id, dist in unlabel2mindist.items():
                abnormals.extend(process(instances[id], "Anomaly"))
                num_outlier_0 += 1
                anomalyOutlierCounter[10] += 1
                if instances[id].type == 'Normal':
                    FP += 1
                    FP_1 += 1
                else:
                    TP += 1
                    TP_1 += 1


        elif res not in normalCores.keys():
            for id in ids:
                confidence = outlier_scores[id] if not math.isnan(outlier_scores[id]) else 1
                if not confidence:
                    num_outlier_0 += 1
                    tmp = 0
                else:
                    tmp = confidence
                anomalyOutlierCounter[math.ceil(tmp * 10)] += 1
                abnormals.extend(process(instances[id], "Anomaly"))
                if instances[id].type == 'Normal':
                    FP += 1
                else:
                    TP += 1

        # Use a threshold to decide whether those instances classified in normal cores are anomaly or not.
        else:
            for id in ids:
                if id >= labelledNum:
                    confidence = outlier_scores[id] if not math.isnan(
                        outlier_scores[id]) else 1
                    if not confidence:
                        num_outlier_0 += 1
                        tmp = 0
                    else:
                        tmp = confidence
                    normalOutlierCounter[math.ceil(tmp * 10)] += 1
                    normals.extend(process(instances[id], "Normal"))
                    if instances[id].type == 'Normal':
                        TN += 1
                    else:
                        FN += 1
    if TP + FP != 0 and TP + FN != 0:
        precision = 100 * TP / (TP + FP)
        recall = 100 * TP / (TP + FN)
        f = 2 * precision * recall / (precision + recall)
        logger.info('Classify finished, precision = %.2f, recall = %.2f, f = %.2f'
                    % (precision, recall, f))
    else:
        logger.info('No TP.')
        precision, recall, f = 0, 0, 0
    abnormals = [x for x in abnormals if " ".join(x.src_events) not in dup['Normal']]
    normals.extend(abnormals)
    logger.info('There are %d instances which outlier scores are 0.' % num_outlier_0)
    return normals, precision, recall, f, num_of_neg1


def upperBound(instances):
    normals = []
    abnormals = []
    for instance in instances:
        if instance.type == 'Normal':
            normals.append(instance)
        else:
            abnormals.append(instance)
    normals.extend(abnormals)
    return normals, 1
