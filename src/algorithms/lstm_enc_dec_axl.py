import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .algorithm import Algorithm
from .cuda_utils import GPUWrapper


class LSTMED(Algorithm, GPUWrapper):
    def __init__(self, hidden_size: int=5, sequence_length: int=30, batch_size: int=20, num_epochs: int=10,
                 n_layers: tuple=(1, 1), use_bias: tuple=(True, True), dropout: tuple=(0, 0),
                 lr: float=0.1, weight_decay: float=1e-4, framework: int=Algorithm.Frameworks.PyTorch, gpu: int=0):
        Algorithm.__init__(self, __name__, 'LSTMED', framework)
        GPUWrapper.__init__(self, gpu)
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lr = lr
        self.weight_decay = weight_decay

        self.lstmed = None

        self.mean = None
        self.cov = None

    def fit(self, X: pd.DataFrame, _):
        X.interpolate(inplace=True)
        data = X.values

        sequences = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = int(0.75 * len(sequences))  # magic number

        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(n_features=X.shape[1], hidden_size=self.hidden_size, batch_size=self.batch_size,
                                   n_layers=self.n_layers, use_bias=self.use_bias, dropout=self.dropout)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lstmed.train()
        for epoch in range(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.lstmed(self.to_var(ts_batch))
                loss = nn.MSELoss(reduce=False)(output, self.to_var(ts_batch.float())).sum()
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(ts_batch.size(0), -1).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

    def predict(self, X: pd.DataFrame):
        prediction_batch_size = 1

        X.interpolate(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=prediction_batch_size, shuffle=False, drop_last=False)

        self.lstmed.batch_size = prediction_batch_size  # (!)
        self.lstmed.eval()

        scores = np.full((self.sequence_length, len(data)), np.nan)
        for idx, ts in enumerate(data_loader):
            output = self.lstmed(self.to_var(ts))

            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -multivariate_normal.logpdf(error.view(1, -1).data.cpu().numpy(), mean=self.mean, cov=self.cov,
                                                allow_singular=True)

            window_elements = np.arange(idx, idx + self.sequence_length, 1)
            scores[idx % self.sequence_length, window_elements] = score

        scores = np.nanmean(scores, axis=0)
        return scores

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self.threshold(score)
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2 * np.nanstd(score)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


class LSTMEDModule(nn.Module, GPUWrapper):
    def __init__(self, n_features: int, hidden_size: int, batch_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple, gpu: int=0):
        super().__init__()
        GPUWrapper.__init__(self, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def init_hidden(self):
        return (self.to_var(torch.zeros(1, self.batch_size, self.hidden_size)),  # first is no of layer.
                self.to_var(torch.zeros(1, self.batch_size, self.hidden_size)))

    def forward(self, ts_batch, return_hidden=False):
        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self.init_hidden()  # initialization with zero
        _, enc_hidden = self.encoder(self.to_var(ts_batch.float()),
                                     enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = (enc_hidden[0], self.to_var(torch.zeros(1, self.batch_size, self.hidden_size)))

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.zeros(ts_batch.size()))
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden) if return_hidden else output
