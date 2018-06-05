'''Adapted from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection'''

import numpy as np

from src.algorithms import LSTMEncDec
from third_party.lstm_enc_dec import preprocess_data
from third_party.lstm_enc_dec import train_predictor
from .algorithm import Algorithm


class EnsembleLSTMEncDec(Algorithm):

    def __init__(self, **kwargs):
        self.name = "Ensemble_LSTM_Enc_Dec"
        train_predictor.set_args(**kwargs)
        self.args = train_predictor.get_args()
        self.best_val_loss = None
        self.train_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.test_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.lstm_enc_dec1 = LSTMEncDec(augment_train_data=False,
                                        prediction_window_size=kwargs["predicition_window_sizes"][0])
        self.lstm_enc_dec2 = LSTMEncDec(augment_train_data=False,
                                        prediction_window_size=kwargs["predicition_window_sizes"][1])
        self.lstm_enc_dec3 = LSTMEncDec(augment_train_data=False,
                                        prediction_window_size=kwargs["predicition_window_sizes"][2])

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
        return self.lstm_enc_dec1.binarize(score)

    def threshold(self, score):
        return self.lstm_enc_dec1.threshold(score)

    def aggregate_scores(self, anomaly_scores1, anomaly_scores2, anomaly_scores3):
        # avg = np.average((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
        # _min = np.min((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
        _max = np.max((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
        return _max
