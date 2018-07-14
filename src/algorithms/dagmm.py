"""Adapted from Daniel Stanley Tan (https://github.com/danieltan07/dagmm)"""
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .algorithm import Algorithm
from .autoencoder import NNAutoEncoder
from .cuda_utils import GPUWrapper


class DAGMMModule(nn.Module, GPUWrapper):
    """Residual Block."""

    def __init__(self, autoencoder, n_gmm=2, latent_dim=3, gpu=0):
        super(DAGMMModule, self).__init__()
        GPUWrapper.__init__(self, gpu)

        self.add_module('autoencoder', autoencoder)

        layers = [
            nn.Linear(latent_dim, 10),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        ]
        self.estimation = nn.Sequential(*layers)
        self.to_device(self.estimation)

        self.register_buffer('phi', self.to_var(torch.zeros(n_gmm)))
        self.register_buffer('mu', self.to_var(torch.zeros(n_gmm, latent_dim)))
        self.register_buffer('cov', self.to_var(torch.zeros(n_gmm, latent_dim, latent_dim)))

    def relative_euclidean_distance(self, a, b, dim=1):
        return (a - b).norm(2, dim=dim) / torch.clamp(a.norm(2, dim=dim), min=1e-10)

    def forward(self, x):
        dec, enc = self.autoencoder(x)

        rec_cosine = F.cosine_similarity(x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1)
        rec_euclidean = self.relative_euclidean_distance(x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1)

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
            phi = Variable(self.phi)
        if mu is None:
            mu = Variable(self.mu)
        if cov is None:
            cov = Variable(self.cov)

        k, d, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + self.to_var(torch.eye(d) * eps)
            pinv = np.linalg.pinv(cov_k.data.numpy())
            cov_inverse.append(Variable(torch.from_numpy(pinv)).unsqueeze(0))

            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            if np.min(eigvals) < 0:
                logging.warning(f'Determinant was negative! Clipping Eigenvalues to 0+epsilon from {np.min(eigvals)}')
            determinant = np.prod(np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None))
            det_cov.append(determinant)

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = Variable(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(self.to_var(phi.unsqueeze(0)) * exp_term / (torch.sqrt(self.to_var(det_cov)) + eps).unsqueeze(0),
                      dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag


class DAGMM(Algorithm, GPUWrapper):
    def __init__(self, num_epochs=10, lambda_energy=0.1, lambda_cov_diag=0.005, lr=1e-2, batch_size=50, gmm_k=3,
                 normal_percentile=80, sequence_length=15, autoencoder_type=NNAutoEncoder, autoencoder_args=None,
                 framework=Algorithm.Frameworks.PyTorch, gpu: int=0):
        window_name = 'withWindow' if sequence_length > 1 else 'withoutWindow'
        Algorithm.__init__(self, __name__, f'DAGMM_{autoencoder_type.__name__}_{window_name}', framework)
        GPUWrapper.__init__(self, gpu)
        self.num_epochs = num_epochs
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.lr = lr
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.gmm_k = gmm_k  # Number of Gaussian mixtures
        self.normal_percentile = normal_percentile  # Up to which percentile data should be considered normal
        self.autoencoder_type = autoencoder_type
        self.autoencoder_args = autoencoder_args or {'gpu': gpu}

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
        total_loss = torch.clamp(total_loss, max=1e7)  # Extremely high loss can cause NaN gradients
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        # if np.array([np.isnan(p.grad.detach().numpy()).any() for p in self.dagmm.parameters()]).any():
        #     import IPython; IPython.embed()
        self.optimizer.step()
        return total_loss, sample_energy, recon_error, cov_diag

    def fit(self, X: pd.DataFrame, _):
        """Learn the mixture probability, mean and covariance for each component k.
        Store the computed energy based on the training data and the aforementioned parameters."""
        X.fillna(0, inplace=True)
        data = X.values
        # Each point is a flattened window and thus has as many features as sequence_length * features
        multi_points = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=multi_points, batch_size=self.batch_size, shuffle=True, drop_last=True)
        hidden_size = max(1, X.shape[1] // 20)
        autoencoder = self.autoencoder_type(n_features=X.shape[1], sequence_length=self.sequence_length,
                                            hidden_size=hidden_size, **self.autoencoder_args)
        self.dagmm = DAGMMModule(autoencoder, n_gmm=self.gmm_k, latent_dim=hidden_size + 2, gpu=self.gpu)
        self.to_device(self.dagmm)
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

        for _ in range(self.num_epochs):
            for input_data in data_loader:
                input_data = self.to_var(input_data)
                self.dagmm_step(input_data.float())

        self.dagmm.eval()
        n = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0
        for input_data in data_loader:
            input_data = self.to_var(input_data)
            _, _, z, gamma = self.dagmm(input_data.float())
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            n += input_data.size(0)

        train_phi = gamma_sum / n
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        train_length = len(data_loader) * self.batch_size + self.sequence_length - 1
        train_energy = np.full((self.sequence_length, train_length), np.nan)
        for i1, ts_batch in enumerate(data_loader):
            ts_batch = self.to_var(ts_batch)
            _, _, z, _ = self.dagmm(ts_batch.float())
            sample_energies, _ = self.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                           size_average=False)

            for i2, sample_energy in enumerate(sample_energies):
                index = i1 * self.batch_size + i2
                window_elements = list(range(index, index + self.sequence_length, 1))
                train_energy[index % self.sequence_length, window_elements] = sample_energy.data.cpu().numpy()
        self.train_energy = np.nanmean(train_energy, axis=0)

    def predict(self, X: pd.DataFrame):
        """Using the learned mixture probability, mean and covariance for each component k, compute the energy on the
        given data."""
        self.dagmm.eval()
        X.fillna(0, inplace=True)
        data = X.values
        multi_points = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=multi_points, batch_size=1, shuffle=False)
        test_energy = np.full((self.sequence_length, len(data)), np.nan)

        for idx, multi_point in enumerate(data_loader):
            _, _, z, _ = self.dagmm(self.to_var(multi_point).float())
            sample_energy, _ = self.dagmm.compute_energy(z, size_average=False)
            window_elements = np.arange(idx, idx + self.sequence_length, 1)
            test_energy[idx % self.sequence_length, window_elements] = sample_energy.data.cpu().numpy()

        test_energy = np.nanmean(test_energy, axis=0)
        combined_energy = np.concatenate([self.train_energy, test_energy], axis=0)

        self._threshold = np.nanpercentile(combined_energy, self.normal_percentile)
        return test_energy

    def threshold(self, score):
        return self._threshold

    def binarize(self, y, threshold=None):
        if threshold is None:
            if self._threshold is not None:
                threshold = self._threshold
            else:
                return np.zeros_like(y)
        return np.where(y > threshold, 1, 0)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
