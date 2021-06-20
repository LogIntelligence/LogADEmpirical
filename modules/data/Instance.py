from collections import Counter
import random, torch
import numpy as np
import hashlib


class Instance:
    def __init__(self, events, src_blk, tag, type, event_ids=None, messages=None, confidence=None, tag_logit=None):
        self.src_events = events
        self.src_messages = messages
        self.src_event_ids = event_ids
        self.src_words = []
        for event in self.src_events:
            self.src_words.extend(event.split())
        self.src_blk = src_blk
        self.tag = tag
        self.type = type
        self._confidence = 0 if confidence is None else confidence
        self.tag_logit = tag_logit
        self.is_anomaly = 1 if self.tag is 'yes' else 0

    def calculate_tf_score(self):
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

    def set_simple_repr(self, repr):
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


class LogEntry(object):
    def __init__(self, log_label, log_dt, log_event, message=""):
        self.log_label = log_label
        self.log_datetime = log_dt
        self.log_event = log_event
        self.log_message = message


class StepLog(object):
    def __init__(self, step_label, step_log_events, step_log_times):
        self.step_label = step_label
        self.step_log_events = step_log_events
        self.step_log_times = step_log_times


def parse_instance(events, blk_id, label, event_ids=None, messages=None, confidence=None):
    events = ['$$'.join(event.split()) for event in events]
    tag = 'yes' if label == 'anomaly' else 'no'
    if confidence is None:
        confidence = 0
    return Instance(events, blk_id, tag, label, event_ids=event_ids, messages=messages, confidence=confidence)