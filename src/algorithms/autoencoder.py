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


class AutoEncoder(Algorithm, GPUWrapper):
    def __init__(self, hidden_size: int=5, sequence_length: int=30, batch_size: int=20, num_epochs: int=10,
                 lr: float=0.1, framework: int=Algorithm.Frameworks.PyTorch, gpu: int=0):
        Algorithm.__init__(self, __name__, 'AutoEncoder', framework)
        GPUWrapper.__init__(self, gpu)
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        self.aed = None

        self.mean = None
        self.cov = None

    def fit(self, X: pd.DataFrame, _):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values

        multi_points = [data[i:i + self.sequence_length].flatten() for i in
                        range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(multi_points))
        split_point = int(0.75 * len(multi_points))  # magic number

        train_loader = DataLoader(dataset=multi_points, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=multi_points, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[split_point:]), pin_memory=True)

        self.aed = AutoEncoderModule(X.shape[1], self.sequence_length, self.hidden_size)
        self.to_device(self.aed)
        optimizer = torch.optim.Adam(self.aed.parameters(), lr=self.lr)

        self.aed.train()
        for epoch in range(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.aed(self.to_var(ts_batch))
                loss = nn.MSELoss(reduce=False)(output, self.to_var(ts_batch.float())).sum()
                self.aed.zero_grad()
                loss.backward()
                optimizer.step()

        self.aed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(ts_batch.size(0), -1).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

    def predict(self, X: pd.DataFrame) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        multi_points = [data[i:i + self.sequence_length].flatten() for i in
                        range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=multi_points, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.aed.eval()
        mvnormal = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True)
        scores = []
        for idx, ts in enumerate(data_loader):
            output = self.aed(self.to_var(ts))

            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(ts.shape[0], -1).data.cpu().numpy())
            scores.append(score)

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        scores = np.pad(scores, (0, self.sequence_length + -len(scores) % self.sequence_length),
                        'constant', constant_values=np.nan)
        scores = np.reshape(scores, (self.sequence_length, -1), 'F')
        scores = np.repeat(scores, self.sequence_length, axis=1)
        scores = np.array([np.roll(row, i) for i, row in enumerate(scores)])
        scores[np.tril_indices(self.sequence_length, k=-1)] = np.nan
        scores = scores[:, :data.shape[0]]
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


class AutoEncoderModule(nn.Module, GPUWrapper):
    def __init__(self, n_features: int, sequence_length: int, hidden_size: int):
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        input_length = n_features * sequence_length

        layers = [nn.Linear(input_length, 60),
                  nn.Tanh(),
                  nn.Linear(60, 30),
                  nn.Tanh(),
                  nn.Linear(30, 10),
                  nn.Tanh(),
                  nn.Linear(10, hidden_size)]

        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

        layers = [nn.Linear(hidden_size, 10),
                  nn.Tanh(),
                  nn.Linear(10, 30),
                  nn.Tanh(),
                  nn.Linear(30, 60),
                  nn.Tanh(),
                  nn.Linear(60, input_length)]

        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, x):
        enc = self._encoder(x)
        dec = self._decoder(enc)

        return dec
