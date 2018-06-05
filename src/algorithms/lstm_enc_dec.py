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


class LSTM_Enc_Dec(Algorithm):

    def __init__(self, **kwargs):
        self.name = "LSTM-Enc-Dec"
        train_predictor.set_args(**kwargs)
        self.args = train_predictor.get_args()
        self.best_val_loss = None
        self.train_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.test_timeseries_dataset: preprocess_data.PickleDataLoad = None

    def _build_model(self, feature_dim):
        self.model = RNNPredictor(rnn_type=self.args.model,
                                  enc_inp_size=feature_dim,
                                  rnn_inp_size=self.args.emsize,
                                  rnn_hid_size=self.args.nhid,
                                  dec_out_size=feature_dim,
                                  nlayers=self.args.nlayers,
                                  dropout=self.args.dropout,
                                  tie_weights=self.args.tied,
                                  res_connection=self.args.res_connection).to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.criterion = nn.MSELoss()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self._build_model(X_train.shape[1])
        train_timeseries_dataset = self._transform_fit_data(X_train, y_train)
        self._fit(train_timeseries_dataset)

    def predict_channel_scores(self, X_test: pd.DataFrame) -> List[np.ndarray]:
        test_timeseries_dataset = self._transform_predict_data(X_test)
        # Anomaly score is returned for each series seperately
        channels_scores, _ = self._predict(test_timeseries_dataset)
        return [x.numpy() for x in channels_scores]

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        channels_scores = self.predict_channel_scores(X_test)
        return np.max(channels_scores, axis=0)

    def binarize(self, score, threshold=None):
        threshold = self.threshold(score)
        score = np.where(np.isnan(score), threshold - 1, score)
        return np.where(score >= threshold, 1, 0)

    """
        Because the algorithm returns only an anomaly score we need to find and
        apply our own threshold for the evaluation. This is done by looking at
        each channel seperately and testing different thresholds (from zero to
        max).
        The resulting amounts of anomalies by threshold are distributed in a
        logarithmic way. We decided to select the threshold by considering
        the mean of all found anomaly amounts.
    """
    def _create_validation_set(self, channels_scores):
        for score in channels_scores:
            maximum = score.max()
            steps = 40
            th = torch.tensor(np.linspace(0, maximum, steps))
            anomalies_by_threshold = np.zeros(len(th))
            for i in range(len(th)):
                anomaly = (score > th[i]).float()
                amount_anomalies = anomaly.sum()
                anomalies_by_threshold[i] = amount_anomalies
            # Find threshold which amount is the closest to the mean of all anomaly amounts
            idx = (np.abs(anomalies_by_threshold - anomalies_by_threshold.mean())).argmin()
            threshold = th[idx]
            logging.info(f'Selecting threshold #{idx}: {threshold}')
            yield np.array(score > threshold, dtype=int)

    def _transform_fit_data(self, X_orig_train, y_orig_train):
        X_train, X_test, y_train, y_test = train_test_split(
            X_orig_train, y_orig_train, test_size=0.25, shuffle=False,
            random_state=42
        )
        self.train_timeseries_dataset = preprocess_data.PickleDataLoad(
            input_data=(X_train, y_train, X_test, y_test),
            augment_train_data=self.args.augment_train_data,
        )
        logging.debug('-'*89)
        logging.debug('Splitting and transforming input data:')
        logging.debug(f'X_orig_train: {X_orig_train.shape}')
        logging.debug(f'y_orig_train: {y_orig_train.shape}')
        logging.debug(f'X_train: {self.train_timeseries_dataset.trainData.shape}')
        logging.debug(f'y_train: {self.train_timeseries_dataset.trainLabel.shape}')
        logging.debug(f'X_val: {self.train_timeseries_dataset.testData.shape}')
        logging.debug(f'y_val: {self.train_timeseries_dataset.testLabel.shape}')
        logging.debug('-'*89)
        return self.train_timeseries_dataset

    def _transform_predict_data(self, X_orig_test):
        self.test_timeseries_dataset = preprocess_data.PickleDataLoad(
            input_data=X_orig_test,
        )
        logging.debug('-'*89)
        logging.debug('Input data:')
        logging.debug(f'X_orig_test: {X_orig_test.shape}')
        logging.debug(f'X_test: {self.test_timeseries_dataset.testData.shape}')
        logging.debug('-'*89)

        return self.test_timeseries_dataset

    def _fit(self, train_timeseries_dataset, start_epoch=1, best_val_loss=0):
        train_dataset = train_timeseries_dataset.batchify(
            self.args, train_timeseries_dataset.trainData, self.args.batch_size)
        test_dataset = train_timeseries_dataset.batchify(
            self.args, train_timeseries_dataset.testData, self.args.eval_batch_size)
        try:
            for epoch in range(start_epoch, self.args.epochs + 1):

                epoch_start_time = time.time()
                train_predictor.train(self.args, self.model, train_dataset, epoch, self.optimizer, self.criterion)

                val_loss = train_predictor.evaluate(self.args, self.model, test_dataset, self.criterion)
                logging.debug('-' * 89)
                logging.debug('| end of epoch {epoch:3d} | time: {time.time():5.2f}s | valid loss {val_loss:5.4f} | ')
                logging.debug('-' * 89)

                if epoch % self.args.save_interval == 0:
                    # Save the model if the validation loss is the best we've seen so far.
                    is_best = val_loss > best_val_loss
                    self.best_val_loss = max(val_loss, best_val_loss)
                    model_dictionary = {'epoch': epoch,
                                        'best_loss': best_val_loss,
                                        'state_dict': self.model.state_dict(),
                                        'optimizer': self.optimizer.state_dict(),
                                        'args': self.args
                                        }
                    self.model.save_checkpoint(model_dictionary, is_best)

        except KeyboardInterrupt:
            logging.warn('-' * 89)
            logging.warn('Exiting from training early')

        # Calculate mean and covariance for each channel's prediction errors, and save them with the trained model
        logging.info('=> calculating mean and covariance')
        means, covs = list(), list()
        train_dataset = train_timeseries_dataset.batchify(self.args, train_timeseries_dataset.trainData, bsz=1)
        for channel_idx in range(self.model.enc_input_size):
            mean, cov = fit_norm_distribution_param(
                self.args, self.model,
                train_dataset[:train_timeseries_dataset.length], channel_idx
            )
            means.append(mean), covs.append(cov)
        model_dictionary = {'epoch': max(epoch, start_epoch),
                            'best_loss': self.best_val_loss,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'args': self.args,
                            'means': means,
                            'covs': covs
                            }
        self.model.save_checkpoint(model_dictionary, True)
        logging.info('-' * 89)

    # For prediction the data is not augmented and not batchified in 64-chunks
    def _predict(self, test_timeseries_dataset):
        # Make train and test data the same size
        test_dataset = test_timeseries_dataset.batchify(self.args, test_timeseries_dataset.testData, bsz=1)
        # In anomaly_detection.py we load the pre-calculated mean and cov
        # from training dataset
        test_timeseries_dataset.trainData = test_timeseries_dataset.testData
        if self.train_timeseries_dataset:
            train_dataset = test_timeseries_dataset.batchify(
                self.args,
                self.train_timeseries_dataset.trainData[:test_timeseries_dataset.length],
                bsz=1
            )
        else:
            # Even in prediction mode the test data is required for calculating
            # the anomaly threshold
            train_dataset = test_dataset
        return anomaly_detection.calc_anomalies(test_timeseries_dataset, train_dataset, test_dataset)

    def threshold(self, score):
        return np.nanmean(score) + 2*np.nanstd(score)
