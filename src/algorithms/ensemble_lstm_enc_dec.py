'''Adapted from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection'''

import logging
import time
from typing import List

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
from third_party.lstm_enc_dec.anomalyDetector import fit_norm_distribution_param
from third_party.lstm_enc_dec import train_predictor
from third_party.lstm_enc_dec import anomaly_detection
from third_party.lstm_enc_dec import preprocess_data
from third_party.lstm_enc_dec.model import RNNPredictor

from .algorithm import Algorithm
from src.algorithms import LSTM_Enc_Dec


class Ensemble_LSTM_Enc_Dec(Algorithm):

    def __init__(self, **kwargs):
        self.name = "Ensemble_LSTM_Enc_Dec"
        train_predictor.set_args(**kwargs)
        self.args = train_predictor.get_args()
        self.best_val_loss = None
        self.train_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.test_timeseries_dataset: preprocess_data.PickleDataLoad = None

