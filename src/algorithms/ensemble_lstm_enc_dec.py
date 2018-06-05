'''Adapted from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection'''

import numpy as np

from src.algorithms import LSTMEncDec
from third_party.lstm_enc_dec import preprocess_data
from third_party.lstm_enc_dec import train_predictor
from .algorithm import Algorithm


class EnsembleLSTMEncDec(Algorithm):

    def __init__(self, prediction_window_size1=5, prediction_window_size2=10, prediction_window_size3=15,
                 aggregation_method="max", **kwargs):
        self.name = "Ensemble_LSTM_Enc_Dec"
        train_predictor.set_args(**kwargs)
        self.args = train_predictor.get_args()
        self.best_val_loss = None
        self.aggregation_method = aggregation_method
        self.train_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.test_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.lstm_enc_dec1 = LSTMEncDec(augment_train_data=False,
                                        prediction_window_size=prediction_window_size1)
        self.lstm_enc_dec2 = LSTMEncDec(augment_train_data=False,
                                        prediction_window_size=prediction_window_size2)
        self.lstm_enc_dec3 = LSTMEncDec(augment_train_data=False,
                                        prediction_window_size=prediction_window_size3)

    def fit(self, X, y):
        self.lstm_enc_dec1.fit(X, y)
        self.lstm_enc_dec2.fit(X, y)
        self.lstm_enc_dec3.fit(X, y)

    def predict(self, X):
        pred1 = self.lstm_enc_dec1.predict(X)
        pred2 = self.lstm_enc_dec2.predict(X)
        pred3 = self.lstm_enc_dec3.predict(X)
        return self.aggregate_scores(pred1, pred2, pred3)

    def binarize(self, score, threshold=None):
        # independent of any external variables
        return self.lstm_enc_dec1.binarize(score)

    def threshold(self, score):
        # independent of any external variables
        return self.lstm_enc_dec1.threshold(score)

    def aggregate_scores(self, anomaly_scores1, anomaly_scores2, anomaly_scores3):
        if self.aggregation_method == "max":
            return np.max((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
        elif self.aggregation_method == "min":
            return np.min((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
        elif self.aggregation_method == "avg":
            np.average((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
