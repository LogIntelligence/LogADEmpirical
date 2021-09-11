import os
import pickle

from logadempirical.logdeep.tools.predict import Predicter
from logadempirical.logdeep.tools.train import Trainer
from logadempirical.logdeep.dataset.vocab import Vocab

from logadempirical.PLELog.pipeline import run_PLELog


def run_plelog(options):
    run_PLELog(options)