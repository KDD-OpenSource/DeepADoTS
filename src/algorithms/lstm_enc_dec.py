'''Adapted from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection'''

import argparse
import time
import torch
import torch.nn as nn
from third_party.lstm_enc_dec import train_predictor
from third_party.lstm_enc_dec import anomaly_detection
from third_party.lstm_enc_dec import preprocess_data
from third_party.lstm_enc_dec.model import RNNPredictor
from torch import optim
from matplotlib import pyplot as plt
from pathlib import Path
from third_party.lstm_enc_dec.anomalyDetector import fit_norm_distribution_param

from .algorithm import Algorithm


class LSTM_Enc_Dec(Algorithm):

    def __init__(self, feature_dim):
        self.args = train_predictor.args
        self.best_val_loss = None

        # Build the model
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

    def fit(self, trainTimeseriesData, start_epoch=1, best_val_loss=0, epochs=train_predictor.args.epochs):
        train_dataset = trainTimeseriesData.batchify(self.args, trainTimeseriesData.trainData, self.args.batch_size)
        test_dataset = trainTimeseriesData.batchify(self.args, trainTimeseriesData.testData, self.args.eval_batch_size)
        gen_dataset = trainTimeseriesData.batchify(self.args, trainTimeseriesData.testData, 1)
        try:
            for epoch in range(start_epoch, epochs + 1):

                epoch_start_time = time.time()
                train_predictor.train(self.args, self.model, train_dataset, epoch, self.optimizer, self.criterion)
                val_loss = train_predictor.evaluate(self.args, self.model, test_dataset, self.criterion)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (
                        time.time() - epoch_start_time), val_loss))
                print('-' * 89)

                train_predictor.generate_output(self.args, epoch, self.model, gen_dataset, trainTimeseriesData, startPoint=1500)

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
    def predict(self, testTimeseriesData):
        # Make train and test data the same size
        train_dataset = testTimeseriesData.batchify(self.args, testTimeseriesData.trainData[:testTimeseriesData.length], bsz=1)
        test_dataset = testTimeseriesData.batchify(self.args, testTimeseriesData.testData, bsz=1)
        return anomaly_detection.calc_anomalies(testTimeseriesData, train_dataset, test_dataset)
