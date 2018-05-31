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


class EnsembleLSTMEncDec(Algorithm):

    def __init__(self, **kwargs):
        self.name = "Ensemble_LSTM_Enc_Dec"
        train_predictor.set_args(**kwargs)
        self.args = train_predictor.get_args()
        self.best_val_loss = None
        self.train_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.test_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.lstm_enc_dec1 = LSTM_Enc_Dec(epochs=1, augment_train_data=True, prediction_window_size=5)
        self.lstm_enc_dec2 = LSTM_Enc_Dec(epochs=1, augment_train_data=True, prediction_window_size=10)
        self.lstm_enc_dec3 = LSTM_Enc_Dec(epochs=1, augment_train_data=True, prediction_window_size=15)

    def fit(self, X, y):
        self.lstm_enc_dec1.fit(X, y)
        self.lstm_enc_dec2.fit(X, y)
        self.lstm_enc_dec3.fit(X, y)


    def predict(self, X):
        pred1 = self.lstm_enc_dec1.predict(X)
        pred2 = self.lstm_enc_dec2.predict(X)
        pred3 = self.lstm_enc_dec3.predict(X)
        return self.eval_anomaly_scores(pred1, pred2, pred3)


    def binarize(self, score, threshold=None):
        LSTM_Enc_Dec.binarize(score)

    def threshold(self, score):
        LSTM_Enc_Dec.threshold(score)


    def eval_anomaly_scores(self, anomaly_scores1, anomaly_scores2, anomaly_scores3):
        #avg = np.average((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
        #_min = np.min((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
        _max = np.max((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)

        return _max
