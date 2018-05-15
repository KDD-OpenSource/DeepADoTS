"""Adapted from Daniel Stanley Tan (https://github.com/danieltan07/dagmm)"""

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "third_party", "dagmm"))

from third_party.dagmm.data_loader import get_loader
from third_party.dagmm.model import *
from third_party.dagmm.utils import *
from .algorithm import Algorithm


class DAGMM(Algorithm):

    def __init__(self, lr=1e-4, num_epochs=200, batch_size=1024, gmm_k=4, lambda_energy=0.1, lambda_cov_diag=0.005,
                 data_path='kdd_cup.npz', log_path='./dagmm/logs', model_save_path='./models/dagmm/', log_step=10,
                 sample_step=194, model_save_step=194):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gmm_k = gmm_k
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.data_path = data_path
        self.log_path = log_path
        self.model_save_path = model_save_path
        self.log_step = log_step
        self.sample_step = sample_step
        self.model_save_step = model_save_step
        self.data_loader = get_loader(self.data_path, batch_size=self.batch_size, mode='train')

    def fit(self, X, y):
        self.dagmm = DaGMM(self.gmm_k)
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.dagmm.cuda()

        self.dagmm.eval()
        self.data_loader.dataset.mode = "train"

        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
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
        train_labels = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=self.train_phi, mu=self.train_mu,
                                                                cov=self.train_cov,
                                                                size_average=False)

            train_energy.append(sample_energy.data.cpu().numpy())

            train_labels.append(labels.numpy())

        self.train_energy = np.concatenate(train_energy, axis=0)

    def reset_grad(self):
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def predict(self, X: np.array):
        self.data_loader.dataset.mode = "test"
        test_energy = []
        test_labels = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_labels.append(labels.numpy())

        test_energy = np.concatenate(test_energy, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        combined_energy = np.concatenate([self.train_energy, test_energy], axis=0)

        thresh = np.percentile(combined_energy, 100 - 20)
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        return gt, pred
