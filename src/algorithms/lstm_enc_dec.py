'''Adapted from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection'''

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

    def __init__(
        self,
        data='lstm_enc_dec',
        filename='chfdb_chf13_45590.pkl',
        model_type='LSTM',  # (RNN_TANH, RNN_RELU, LSTM, GRU)
        augment_train_data=True,
        emsize=32,  # size of rnn input features
        nhid=32,  # number of hidden units per layer
        nlayers=2,  # number of layers
        res_connection=False,  # residual connection
        learning_rate=0.0002,  # initial learning rate for Adam
        weight_decay=1e-4,
        gradient_clip=10,  # to avoid exploding gradients
        epochs=20,
        batch_size=64,
        eval_batch_size=64,
        seq_length=50,
        dropout=0.2,
        tied=False,  # tie the word embedding and softmax weights (deprecated)
        seed=1111,
        device='cpu',  # cuda/cpu
        log_interval=10,
        save_interval=10,
        resume=False,  # use checkpoint model parameters as initial parameters
        pretrained=False,  # use checkpoint model parameters and do not train anymore
        prediction_window_size=10
    ):
        super().__init__(__name__, "LSTM-Enc-Dec")
        self.data = data
        self.filename = filename
        self.model = None
        self.model_type = model_type
        self.augment_train_data = augment_train_data
        self.emsize = emsize
        self.nhid = nhid
        self.nlayers = nlayers
        self.res_connection = res_connection
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.seq_length = seq_length
        self.dropout = dropout
        self.tied = tied
        self.device = device
        self.seed = seed
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume = resume
        self.pretrained = pretrained
        self.prediction_window_size = prediction_window_size
        self.best_val_loss = None
        self.train_timeseries_dataset: preprocess_data.PickleDataLoad = None
        self.test_timeseries_dataset: preprocess_data.PickleDataLoad = None

        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def _build_model(self, feature_dim):
        self.model = RNNPredictor(rnn_type=self.model_type,
                                  enc_inp_size=feature_dim,
                                  rnn_inp_size=self.emsize,
                                  rnn_hid_size=self.nhid,
                                  dec_out_size=feature_dim,
                                  nlayers=self.nlayers,
                                  dropout=self.dropout,
                                  tie_weights=self.tied,
                                  res_connection=self.res_connection).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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

    def threshold(self, score):
        return np.nanmean(score) + 2*np.nanstd(score)

    def _transform_fit_data(self, X_orig_train, y_orig_train):
        X_train, X_test, y_train, y_test = train_test_split(
            X_orig_train, y_orig_train, test_size=0.25, shuffle=False,
            random_state=42
        )
        self.train_timeseries_dataset = preprocess_data.PickleDataLoad(
            input_data=(X_train, y_train, X_test, y_test),
            augment_train_data=self.augment_train_data,
        )
        self.logger.debug('-'*89)
        self.logger.debug('Splitting and transforming input data:')
        self.logger.debug(f'X_orig_train: {X_orig_train.shape}')
        self.logger.debug(f'y_orig_train: {y_orig_train.shape}')
        self.logger.debug(f'X_train: {self.train_timeseries_dataset.trainData.shape}')
        self.logger.debug(f'y_train: {self.train_timeseries_dataset.trainLabel.shape}')
        self.logger.debug(f'X_val: {self.train_timeseries_dataset.testData.shape}')
        self.logger.debug(f'y_val: {self.train_timeseries_dataset.testLabel.shape}')
        self.logger.debug('-'*89)
        return self.train_timeseries_dataset

    def _transform_predict_data(self, X_orig_test):
        self.test_timeseries_dataset = preprocess_data.PickleDataLoad(
            input_data=X_orig_test,
        )
        self.logger.debug('-'*89)
        self.logger.debug('Input data:')
        self.logger.debug(f'X_orig_test: {X_orig_test.shape}')
        self.logger.debug(f'X_test: {self.test_timeseries_dataset.testData.shape}')
        self.logger.debug('-'*89)

        return self.test_timeseries_dataset

    def _fit(self, train_timeseries_dataset, start_epoch=1, best_val_loss=0):
        train_dataset = train_timeseries_dataset.batchify(
            self.device, train_timeseries_dataset.trainData, self.batch_size)
        test_dataset = train_timeseries_dataset.batchify(
            self.device, train_timeseries_dataset.testData, self.eval_batch_size)
        try:
            for epoch in range(start_epoch, self.epochs + 1):

                epoch_start_time = time.time()
                train_predictor.train(self.model, train_dataset, epoch, self.optimizer,
                                      self.criterion, self.batch_size, self.seq_length,
                                      self.log_interval, self.gradient_clip)

                val_loss = train_predictor.evaluate(self.model, test_dataset, self.criterion,
                                                    self.batch_size, self.eval_batch_size,
                                                    self.seq_length)
                self.logger.debug('-' * 89)
                run_time = time.time() - epoch_start_time
                self.logger.debug(f'| end of epoch {epoch:3d} | time: {run_time:5.2f}s | valid loss {val_loss:5.4f} | ')
                self.logger.debug('-' * 89)

                if epoch % self.save_interval == 0:
                    # Save the model if the validation loss is the best we've seen so far.
                    is_best = val_loss > best_val_loss
                    self.best_val_loss = max(val_loss, best_val_loss)
                    self._save_checkpoint(epoch, best_val_loss, is_best)
        except KeyboardInterrupt:
            self.logger.warning('-' * 89)
            self.logger.warning('Exiting from training early')

        # Calculate mean and covariance for each channel's prediction errors, and save them with the trained model
        self.logger.info('=> calculating mean and covariance')
        means, covs = list(), list()
        train_dataset = train_timeseries_dataset.batchify(
            self.device, train_timeseries_dataset.trainData, bsz=1)
        for channel_idx in range(self.model.enc_input_size):
            mean, cov = fit_norm_distribution_param(
                self.model, train_dataset[:train_timeseries_dataset.length],
                self.prediction_window_size, self.device, channel_idx,
            )
            means.append(mean), covs.append(cov)
        self._save_checkpoint(epoch, self.best_val_loss, means=means, covs=covs)
        self.logger.info('-' * 89)

    # For prediction the data is not augmented and not batchified in 64-chunks
    def _predict(self, test_timeseries_dataset):
        # Make train and test data the same size
        test_dataset = test_timeseries_dataset.batchify(
            self.device, test_timeseries_dataset.testData, bsz=1)
        # In anomaly_detection.py we load the pre-calculated mean and cov from training dataset
        test_timeseries_dataset.trainData = test_timeseries_dataset.testData
        if self.train_timeseries_dataset:
            train_dataset = test_timeseries_dataset.batchify(
                self.device,
                self.train_timeseries_dataset.trainData[:test_timeseries_dataset.length],
                bsz=1
            )
        else:
            # Even in prediction mode the test data is required for calculating
            # the anomaly threshold
            train_dataset = test_dataset
        return anomaly_detection.calc_anomalies(
            test_timeseries_dataset, train_dataset, test_dataset, self.device,
            self.data, self.filename)

    def _save_checkpoint(self, epoch, best_loss, save_model=True, means=None, covs=None):
        # For reproducibility a set of arguments is stored which will be reused during prediction
        stored_args = {
            'seed': self.seed,
            'model_type': self.model_type,
            'emsize': self.emsize,
            'nhid': self.nhid,
            'nlayers': self.nlayers,
            'dropout': self.dropout,
            'res_connection': self.res_connection,
            'prediction_window_size': self.prediction_window_size,
            'device': self.device,
        }
        model_dictionary = {
            'epoch': epoch,
            'best_loss': best_loss,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': stored_args,
            'means': means,
            'covs': covs,
        }
        self.model.save_checkpoint(model_dictionary, save_model, self.data, self.filename)
