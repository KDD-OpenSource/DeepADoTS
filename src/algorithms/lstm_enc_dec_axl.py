import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .algorithm import Algorithm


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class LSTMED(Algorithm):
    def __init__(self, hidden_size: int=5, sequence_length: int= 30, batch_size: int=20, epochs: int=10,
                 n_layers: tuple = (1, 1), use_bias: tuple=(True, True), dropout: tuple=(0, 0),
                 lr: float=0.1, weight_decay: float=1e-4, criterion=nn.MSELoss()):
        super().__init__('LSTMED')
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion

        self.lstmed = None

    def fit(self, X: pd.DataFrame, _):
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.lstmed = LSTMEDModule(n_features=X.shape[1], hidden_size=self.hidden_size, batch_size=self.batch_size,
                                   n_layers=self.n_layers, use_bias=self.use_bias, dropout=self.dropout)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lstmed.train()
        for epoch in range(self.epochs):
            logging.debug(f'Epoch {epoch}/{self.epochs}.')
            for ts_batch in data_loader:
                output = self.lstmed(to_var(ts_batch))

                loss = self.criterion(output, ts_batch.float())

                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame):
        prediction_batch_size = 1

        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=prediction_batch_size, shuffle=False, drop_last=True)

        self.lstmed.batch_size = prediction_batch_size  # (!)
        self.lstmed.eval()
        errors = [np.nan]*(self.sequence_length-1)
        for ts in data_loader:
            output = self.lstmed(ts)
            errors.append(np.float(self.criterion(output, ts.float())))
        logging.debug('LSTMED just returning MSE. Does not comply with the paper.')
        return np.array(errors)

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self.threshold(score)
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2*np.nanstd(score)


class LSTMEDModule(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, batch_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_size),  # first is no of layer.
                torch.zeros(1, self.batch_size, self.hidden_size))

    def forward(self, ts_batch, return_hidden=False):
        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self.init_hidden()  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = (enc_hidden[0], torch.zeros(1, self.batch_size, self.hidden_size))

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = torch.zeros(ts_batch.size())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden) if return_hidden else output
