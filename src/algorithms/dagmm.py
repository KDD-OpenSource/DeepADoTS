"""Adapted from Daniel Stanley Tan (https://github.com/danieltan07/dagmm)"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from .algorithm import Algorithm

sys.path.append(os.path.join(os.getcwd(), "third_party", "dagmm"))
#from third_party.dagmm.model import DaGMM  # noqa
#from third_party.dagmm.utils import to_var  # noqa


class CustomDataLoader(object):
    """Wrap the given features so they can be put into a torch DataLoader"""

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return np.float32(self.X[index])


class DAGMM(Algorithm):

    def __init__(self, lr=1e-4, batch_size=1024, gmm_k=4, normal_percentile=80):
        self.lr = lr
        self.batch_size = batch_size
        self.gmm_k = gmm_k  # Number of Gaussian mixtures
        self.normal_percentile = normal_percentile  # Up to which percentile data should be consideres normal

    def _reset_grad(self):
        self.dagmm.zero_grad()

    def fit(self, X, _):
        """Learn the mixture probability, mean and covariance for each component k.
        Store the computed energy based on the training data and the aforementioned parameters."""
        data_loader = DataLoader(dataset=CustomDataLoader(X), batch_size=self.batch_size, shuffle=True)

        self.dagmm = DAGMM(self.gmm_k)
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.dagmm.cuda()

        self.dagmm.eval()

        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, input_data in enumerate(data_loader):
            input_data = to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            N += input_data.size(0)

        self.train_phi = gamma_sum / N
        self.train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        self.train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        train_energy = []
        for it, input_data in enumerate(data_loader):
            input_data = to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=self.train_phi, mu=self.train_mu,
                                                                cov=self.train_cov, size_average=False)
            train_energy.append(sample_energy.data.cpu().numpy())

        self.train_energy = np.concatenate(train_energy, axis=0)

    def predict(self, X):
        """Using the learned mixture probability, mean and covariance for each component k, compute the energy on the
        given data and label an anomaly if it is outside of the `self.normal_percentile` percentile."""
        data_loader = DataLoader(dataset=CustomDataLoader(X), batch_size=self.batch_size, shuffle=False)

        test_energy = []
        for it, input_data in enumerate(data_loader):
            input_data = to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=self.train_phi, mu=self.train_mu,
                                                                cov=self.train_cov, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())

        test_energy = np.concatenate(test_energy, axis=0)
        combined_energy = np.concatenate([self.train_energy, test_energy], axis=0)

        thresh = np.percentile(combined_energy, self.normal_percentile)
        return (test_energy > thresh).astype(int)
