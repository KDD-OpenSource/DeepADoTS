import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .algorithm import Algorithm
from .lstm_enc_dec_axl import LSTMEDModule


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class LSTMEDGMM(Algorithm):
    def __init__(self, hidden_size: int=1, sequence_length: int=30, batch_size: int=70, epochs: int=10,
                 n_layers: tuple=(1, 1), use_bias: tuple=(True, True), dropout: tuple=(0, 0),
                 lr: float=0.1, weight_decay: float=1e-4,
                 lambda_energy: float=0.1, lambda_cov_diag: float=5e-3, gmm_k: int=4, normal_percentile: int=80):
        super().__init__("LSTMEDGMM")
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lr = lr
        self.weight_decay = weight_decay

        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.gmm_k = gmm_k  # Number of Gaussian mixtures
        self.normal_percentile = normal_percentile  # Up to which percentile data should be considered normal
        self.lstmedgmm, self.optimizer, self.train_energy, self._threshold = None, None, None, None

    def fit(self, X: pd.DataFrame, _):
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.lstmedgmm = LSTMEDGMMModule(n_features=X.shape[1], hidden_size=self.hidden_size,
                                         batch_size=self.batch_size, n_gmm=self.gmm_k,
                                         n_layers=self.n_layers, use_bias=self.use_bias, dropout=self.dropout)
        self.optimizer = torch.optim.Adam(self.lstmedgmm.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lstmedgmm.train()
        for epoch in range(self.epochs):
            logging.debug(f'Epoch {epoch}/{self.epochs}.')
            for ts_batch in data_loader:
                output, enc_hidden, z, gamma = self.lstmedgmm(to_var(ts_batch))

                total_loss, sample_energy, recon_error, cov_diag = \
                    self.lstmedgmm.loss_function(ts_batch, output, z, gamma, self.lambda_energy, self.lambda_cov_diag)

                self.lstmedgmm.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstmedgmm.parameters(), 5)  # check whether this is useful
                self.optimizer.step()

        self.lstmedgmm.eval()
        n = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0
        for ts_batch in data_loader:
            _, _, z, gamma = self.lstmedgmm(to_var(ts_batch))
            phi, mu, cov = self.lstmedgmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            n += ts_batch.size(0)

        train_phi = gamma_sum / n
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        train_energy = []
        for ts_batch in data_loader:
            _, _, z, _ = self.lstmedgmm(to_var(ts_batch))
            sample_energy, _ = self.lstmedgmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                             size_average=False)
            train_energy.append(sample_energy.data.cpu().numpy())

        self.train_energy = np.concatenate(train_energy, axis=0)

    def predict(self, X: pd.DataFrame):
        prediction_batch_size = 1

        X = X.dropna()
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=prediction_batch_size, shuffle=False, drop_last=True)

        self.lstmedgmm.batch_size = prediction_batch_size  # (!)
        self.lstmedgmm.eval()
        test_energy = [[np.nan]]*(self.sequence_length - 1)
        for ts in data_loader:
            _, _, z, _ = self.lstmedgmm(to_var(ts))
            sample_energy, _ = self.lstmedgmm.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())

        test_energy = np.concatenate(test_energy, axis=0)
        combined_energy = np.concatenate([self.train_energy, test_energy], axis=0)

        self._threshold = np.nanpercentile(combined_energy, self.normal_percentile)
        if np.isnan(self._threshold):
            raise Exception("Threshold is NaN!")

        return test_energy

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self._threshold
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score > threshold, 1, 0)

    def threshold(self, score):
        return self._threshold


class LSTMEDGMMModule(LSTMEDModule):
    def __init__(self, n_features: int, hidden_size: int, batch_size: int, n_gmm: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple):
        super().__init__(n_features, hidden_size, batch_size, n_layers, use_bias, dropout)
        self.n_gmm = n_gmm

        latent_size = self.hidden_size + 2  # there are 2 distance measures right now
        layers = [
            nn.Linear(latent_size, 10),  # a bit arbitrary
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, self.n_gmm),
            nn.Softmax(dim=1)
        ]
        self.estimation = nn.Sequential(*layers)
        self.register_buffer("phi", torch.zeros(self.n_gmm))
        self.register_buffer("mu", torch.zeros(self.n_gmm, latent_size))
        self.register_buffer("cov", torch.zeros(self.n_gmm, latent_size, latent_size))
        self.phi, self.mu, self.cov = None, None, None

    def forward(self, ts_batch, **kwargs):
        output, enc_hidden = super().forward(ts_batch, return_hidden=True)

        hidden_state = enc_hidden[0]
        euclidean = nn.PairwiseDistance()(output, ts_batch.float()).sum(1)  # sum of dists in all dimensions, care!
        cosine = nn.CosineSimilarity()(output, ts_batch.float()).sum(1)  # use only one. review multivariate cases

        # Concatenate hidden_state representation and distance measure
        z = torch.cat([hidden_state.squeeze(0), euclidean.unsqueeze(1), cosine.unsqueeze(1)], dim=1)
        gamma = self.estimation(z)
        return output, enc_hidden, z, gamma

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = nn.MSELoss()(x_hat, x.float())
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag

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
