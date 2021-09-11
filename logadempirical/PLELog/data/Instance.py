from collections import Counter
import random, torch
import numpy as np
import hashlib






class Instance:
    def __init__(self, events, src_blk, tag, type, confidence=None, tag_logit=None):
        self.src_events = events
        self.src_words = []
        for event in self.src_events:
            self.src_words.extend(event.split())
        self.src_blk = src_blk
        self.tag = tag
        self.type = type
        self._confidence = 0 if confidence is None else confidence
        self.tag_logit = tag_logit

    def calculateTFScore(self):
        self.tf = {}
        vocab = set(self.src_words)
        total = len(self.src_words)
        counter = Counter()
        for event in self.src_events:
            words = event.split()
            for token in words:
                counter[token] += 1
        for token in vocab:
            self.tf[token] = counter[token] / total

    def setSimpleRepr(self, repr):
        self._repr = repr

    @property
    def repr(self):
        return self._repr

    def __str__(self):
        ## print source words
        output = ''
        output += ' '.join(self.src_events) + '\n'
        output += str(self.src_blk) + ','
        if self.tag == 'yes':
            output = output + 'Anomaly,' + str(self.confidence)
        else:
            output = output + 'Normal,' + str(self.confidence)
        if self.tag_logit:
            output += ',' + ','.join([str(x) for x in self.tag_logit])
        return output + '\n'

    @property
    def src_len(self):
        return len(self.src_words)

    @property
    def confidence(self):
        return self._confidence

    def __hash__(self):
        return hash(hashlib.md5(' '.join(self.src_events).encode('utf-8')).hexdigest())

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class BGL_Log(object):
    def __init__(self, log_label, log_dt, log_event):
        self.log_label = log_label
        self.log_dt = log_dt
        self.log_event = log_event


class Step_log(object):
    def __init__(self, step_label, step_log_events, step_log_times):
        self.step_label = step_label
        self.step_log_events = step_log_events
        self.step_log_times = step_log_times


# class Window_log(object):
#     def __init__(self, src_events, tag, type):
#         self.type = type
#         self.tag = tag
#         self.src_events = src_events
#         self.src_words = []
#         for event in self.src_events:
#             self.src_words.extend(event.split())
#
#     def setSimpleRepr(self, repr):
#         self._repr = repr
#
#     @property
#     def repr(self):
#         return self._repr
#
#     def __str__(self):
#         ## print source words
#         output = ''
#         temp = []
#         for event in self.src_events:
#             temp.append('$$'.join(event.split()))
#         output += ' '.join(temp) + ' '
#         # for event in self.src_events:
#         #     output += (str(event) + ' ')
#         if self.tag == 'yes':
#             output = output + 'Anomaly\n'
#         else:
#             output = output + 'Normal\n'
#         return output
#
#     @property
#     def src_len(self):
#         return len(self.src_words)
#
#     def __hash__(self):
#         return hash(self.__str__())


class HDbscan_Instance():
    def __init__(self, events, tag, repr, id, cluster, outlier_score, type):
        self.events = events
        self.repr = repr
        self.id = id
        self.cluster = cluster
        self.outlier = outlier_score
        # one of ['Normal','Anomaly']
        self.tag = tag
        # one of ['labelled','unlabelled']
        self.type = type

    def __str__(self):
        output = ''
        output += ' '.join(self.events) + '\n'
        output += ' '.join([str(x) for x in self.repr.tolist()]) + '\n'
        output += ','.join([str(self.id), str(self.cluster), str(self.outlier), str(self.tag), str(self.type)]) + '\n'
        return output


def parseHDbscanInstance(context):
    events = context[0].split()
    repr = np.asarray(context[1].split(), dtype=np.float)
    id, cluster, outlier, tag, type = context[2].split(',')
    return HDbscan_Instance(events, tag, repr, id, cluster, outlier, type)


def parseInstance(events, blk_ID, label, confidence=None):
    events = ['$$'.join(event.split()) for event in events]
    tag = 'yes' if label == 'Anomaly' else 'no'
    if confidence is None:
        confidence = 0
    return Instance(events, blk_ID, tag, label, confidence)


def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')


def printInstance(output, inst):
    output.write(str(inst) + '\n')
