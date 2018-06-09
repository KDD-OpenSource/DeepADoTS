"""Adapted from Daniel Stanley Tan (https://github.com/danieltan07/dagmm)"""
import abc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .algorithm import Algorithm


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class CustomDataLoader(object):
    """Wrap the given features so they can be put into a torch DataLoader"""

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return np.float32(self.X[index])


class AutoEncoder():

    @abc.abstractmethod
    def __call__(self, x):
        """Run autoencoder, return (decoded, encoded)"""


class NNAutoEncoder(AutoEncoder):

    def __init__(self, n_features=118, hidden_size=1):
        layers = []
        layers += [nn.Linear(n_features, 60)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(60, 30)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(30, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(10, hidden_size)]

        self._encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(hidden_size, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(10, 30)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(30, 60)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(60, n_features)]

        self._decoder = nn.Sequential(*layers)

    def __call__(self, x):
        enc = self._encoder(x)
        dec = self._decoder(x)

        return dec, enc


class LSTMAutoEncoder(AutoEncoder):
    """Autoencoder with Recurrent module. Inspired by LSTM-EncDec"""

    def __init__(self, n_features=118, hidden_size=1, dropout=0.5, layers=2):

        layers = [
            nn.Linear(n_features, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, hidden_size),
            nn.Dropout(dropout),
            nn.LSTM(hidden_size, 1, layers=2, dropout=dropout)
        ]
        self._encoder = nn.Sequential(*layers)

        layers = [
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, n_features)
        ]
        self._decoder = nn.Sequential(*layers)

    def __call__(self, x):
        output, hidden = self._encoder(x)
        decoded = self._decoder(output)

        return decoded, hidden


class DAGMM_Module(nn.Module):
    """Residual Block."""

    def __init__(self, autoencoder, n_gmm=2, latent_dim=3):
        super(DAGMM_Module, self).__init__()

        self.autoencoder = autoencoder

        layers = []
        layers += [nn.Linear(latent_dim, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(10, n_gmm)]
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / torch.clamp(a.norm(2, dim=1), min=1e-10)

    def forward(self, x):
        dec, enc = self.autoencoder(x)

        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)

        # Concatenate latent representation, cosine similarity and relative Euclidean distance between x and dec(enc(x))
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, d, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(d) * eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy() * (2 * np.pi)))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag


class DAGMM(Algorithm):
    def __init__(self, num_epochs=5, lambda_energy=0.1, lambda_cov_diag=0.005, lr=1e-4, batch_size=700, gmm_k=3,
                 normal_percentile=80):
        self.name = "DAGMM"
        self.num_epochs = num_epochs
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.lr = lr
        self.batch_size = batch_size
        self.gmm_k = gmm_k  # Number of Gaussian mixtures
        self.normal_percentile = normal_percentile  # Up to which percentile data should be considered normal
        self.dagmm, self.optimizer, self.train_energy, self._threshold = None, None, None, None

    def reset_grad(self):
        self.dagmm.zero_grad()

    def dagmm_step(self, input_data):
        self.dagmm.train()
        enc, dec, z, gamma = self.dagmm(input_data)
        total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma,
                                                                                    self.lambda_energy,
                                                                                    self.lambda_cov_diag)
        self.reset_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()
        return total_loss, sample_energy, recon_error, cov_diag

    def fit(self, X: pd.DataFrame, _):
        """Learn the mixture probability, mean and covariance for each component k.
        Store the computed energy based on the training data and the aforementioned parameters."""
        X = X.dropna()
        data_loader = DataLoader(dataset=CustomDataLoader(X.values), batch_size=self.batch_size, shuffle=False)
        self.dagmm = DAGMM_Module(autoencoder=NNAutoEncoder(n_features=X.shape[1]), n_gmm=self.gmm_k)
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)
        self.dagmm.eval()

        for _ in range(self.num_epochs):
            for input_data in data_loader:
                input_data = to_var(input_data)
                self.dagmm_step(input_data)

        n = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for input_data in data_loader:
            input_data = to_var(input_data)
            _, _, z, gamma = self.dagmm(input_data)
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            n += input_data.size(0)

        train_phi = gamma_sum / n
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        train_energy = []
        for input_data in data_loader:
            input_data = to_var(input_data)
            _, _, z, _ = self.dagmm(input_data)
            sample_energy, _ = self.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                         size_average=False)
            train_energy.append(sample_energy.data.cpu().numpy())

        self.train_energy = np.concatenate(train_energy, axis=0)

    def predict(self, X: pd.DataFrame):
        """Using the learned mixture probability, mean and covariance for each component k, compute the energy on the
        given data."""
        X = X.dropna()
        test_energy = []
        data_loader = DataLoader(dataset=CustomDataLoader(X.values), batch_size=self.batch_size, shuffle=False)
        for input_data in data_loader:
            input_data = to_var(input_data)
            _, _, z, _ = self.dagmm(input_data)
            sample_energy, _ = self.dagmm.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())

        test_energy = np.concatenate(test_energy, axis=0)
        combined_energy = np.concatenate([self.train_energy, test_energy], axis=0)
        self._threshold = np.percentile(combined_energy, self.normal_percentile)
        if np.isnan(self._threshold):
            raise Exception("Threshold is NaN")
        return test_energy

    def threshold(self, score):
        return self._threshold

    def binarize(self, y, threshold=None):
        if threshold is None:
            threshold = self._threshold
        return np.where(y > threshold, 1, 0)
