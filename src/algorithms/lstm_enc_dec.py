'''Adapted from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection'''

import argparse
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from third_party.lstm_enc_dec.anomalyDetector import fit_norm_distribution_param
from third_party.lstm_enc_dec import train_predictor
from third_party.lstm_enc_dec import anomaly_detection
from third_party.lstm_enc_dec import preprocess_data
from third_party.lstm_enc_dec.model import RNNPredictor

from .algorithm import Algorithm


class LSTM_Enc_Dec(Algorithm):

    def __init__(self, **kwargs):
        train_predictor.set_args(**kwargs)
        self.args = train_predictor.get_args()
        self.best_val_loss = None
        self.trainTimeseriesData = None
        self.testTimeseriesData = None

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

    # X_train is a DataFrame (e.g. 1000x4), y_train is a Series (e.g. 1000)
    def fit(self, X_train, y_train):
        self._build_model(X_train.shape[1])
        trainTimeseriesData = self.transform_fit_data(X_train, y_train)
        self.intern_fit(trainTimeseriesData)

    # X_test is a DataFrame (e.g. 200x4)
    # Returns anomaly score as Series (e.g. 200)
    def predict(self, X_test):
        testTimeseriesData = self.transform_predict_data(X_test)
        # Anomaly score is returned for each series seperately
        channels_scores, _ = self.intern_predict(testTimeseriesData)
        channels_scores = [x.numpy() for x in channels_scores]
        # plt.plot(channels_scores[0])
        # plt.plot(channels_scores[1])
        # plt.plot(channels_scores[2])
        # plt.plot(channels_scores[3])
        # plt.savefig('scores.png')
        # plt.close()
        binary_decisions = np.array(list(self.find_fitting_threshold(channels_scores)))
        return np.max(binary_decisions, axis=0)

    def find_fitting_threshold(self, channels_scores):
        plot_xmax = max([x.max() for x in channels_scores])
        for j, score in enumerate(channels_scores):
            maximum = score.max()
            # Sample thresholds logarithmically
            # The sampled thresholds are logarithmically spaced between: math:`10 ^ {start}` and: math:`10 ^ {end}`.
            # th = torch.logspace(0, torch.log10(torch.tensor(float(maximum))), threshold_checks).to(self.args.device)
            th = torch.tensor(np.linspace(0, maximum, 20))
            anomalies_by_threshold = np.zeros(len(th))
            for i in range(len(th)):
                anomaly = (score > th[i]).float()
                amount_anomalies = anomaly.sum()
                anomalies_by_threshold[i] = amount_anomalies
                diff = anomalies_by_threshold[max(i-1, 0)] - anomalies_by_threshold[i]

            p = plt.plot(th.numpy(), anomalies_by_threshold)
            threshold = np.median(anomalies_by_threshold) + anomalies_by_threshold.std() / 2
            # plt.hlines(threshold, 0, plot_xmax, color=p[-1].get_color(), linestyles='dashed')
            yield np.array(score > threshold, dtype=int)
        # plt.ylabel('Amount of anomalies')
        # plt.xlabel('Threshold')
        # plt.savefig('anomalies_by_threshold.png')
        # plt.close()

    def transform_fit_data(self, X_orig_train, y_orig_train):
        X_train, X_test, y_train, y_test = train_test_split(X_orig_train, y_orig_train, test_size=0.25, shuffle=False, random_state=42)
        self.trainTimeseriesData = preprocess_data.PickleDataLoad(
            input_data=(X_train, y_train, X_test, y_test),
            augment_train_data=self.args.augment_train_data,
        )
        print('-'*89)
        print('Splitting and transforming input data:')
        print('X_orig_train', X_orig_train.shape)
        print('y_orig_train', y_orig_train.shape)
        print('X_train', self.trainTimeseriesData.trainData.shape)
        print('y_train', self.trainTimeseriesData.trainLabel.shape)
        print('X_val', self.trainTimeseriesData.testData.shape)
        print('y_val', self.trainTimeseriesData.testLabel.shape)
        print('-'*89)
        return self.trainTimeseriesData

    def transform_predict_data(self, X_orig_test):
        # TODO: adjust anomaly_detection to not calculate precision, recall and stuff
        self.testTimeseriesData = preprocess_data.PickleDataLoad(
            input_data=X_orig_test,
        )
        print('-'*89)
        print('Input data:')
        print('X_orig_test', X_orig_test.shape)
        print('X_test', self.testTimeseriesData.testData.shape)
        print('-'*89)

        return self.testTimeseriesData

    def intern_fit(self, trainTimeseriesData, start_epoch=1, best_val_loss=0):
        train_dataset = trainTimeseriesData.batchify(self.args, trainTimeseriesData.trainData, self.args.batch_size)
        test_dataset = trainTimeseriesData.batchify(self.args, trainTimeseriesData.testData, self.args.eval_batch_size)
        gen_dataset = trainTimeseriesData.batchify(self.args, trainTimeseriesData.testData, 1)
        try:
            for epoch in range(start_epoch, self.args.epochs + 1):

                epoch_start_time = time.time()
                train_predictor.train(self.args, self.model, train_dataset, epoch, self.optimizer, self.criterion)
                val_loss = train_predictor.evaluate(self.args, self.model, test_dataset, self.criterion)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (
                        time.time() - epoch_start_time), val_loss))
                print('-' * 89)

                # TODO: Only plots figures - doesn't work right now because of start and endPoint (what is gen_data)
                validation_length = len(gen_dataset)
                train_predictor.generate_output(
                    self.args, epoch, self.model, gen_dataset, trainTimeseriesData,
                    startPoint=int(validation_length / 4), endPoint=validation_length - 1
                )

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
            print('-' * 89)
            print('Exiting from training early')

        # Calculate mean and covariance for each channel's prediction errors, and save them with the trained model
        print('=> calculating mean and covariance')
        means, covs = list(), list()
        train_dataset = trainTimeseriesData.batchify(self.args, trainTimeseriesData.trainData, bsz=1)
        for channel_idx in range(self.model.enc_input_size):
            mean, cov = fit_norm_distribution_param(self.args, self.model, train_dataset[:trainTimeseriesData.length], channel_idx)
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
        print('-' * 89)

    # For prediction the data is not augmented and not batchified in 64-chunks
    def intern_predict(self, testTimeseriesData):
        # Make train and test data the same size
        test_dataset = testTimeseriesData.batchify(self.args, testTimeseriesData.testData, bsz=1)
        if self.trainTimeseriesData:
            train_dataset = testTimeseriesData.batchify(self.args, self.trainTimeseriesData.trainData[:testTimeseriesData.length], bsz=1)
        else:
            # Even in prediction mode the test data is required for calculating
            # the anomaly threshold
            train_dataset = test_dataset
            testTimeseriesData.trainData = testTimeseriesData.testData
        return anomaly_detection.calc_anomalies(testTimeseriesData, train_dataset, test_dataset)
