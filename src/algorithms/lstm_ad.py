from . import Algorithm

import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch.autograd import Variable


class LSTMSequence(torch.nn.Module):
    def __init__(self, d, len_in=1, len_out=10):
        super().__init__()
        self.d = d  # input and output feature dimensionality
        self.len_out = len_out
        self.hidden_size1 = 32
        self.hidden_size2 = 32
        self.lstm1 = torch.nn.LSTMCell(d * len_in, self.hidden_size1)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_size1, self.hidden_size2)
        self.linear = torch.nn.Linear(self.hidden_size2, d * len_out)

    def forward(self, input):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.hidden_size1).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.hidden_size1).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), self.hidden_size2).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), self.hidden_size2).double(), requires_grad=False)
        for input_t in input.chunk(input.size(1), dim=1):
            h_t, c_t = self.lstm1(input_t.squeeze(dim=1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze()  # stack (n, d * len_out) outputs in time dimensionality (dim=1)

        return outputs.view(input.size(0), input.size(1), self.d, self.len_out)


class LSTMAD(Algorithm):
    """ LSTM-AD implementation using PyTorch.
    The interface of the class is sklearn-like.
    """

    def __init__(self, len_out=10, num_epochs=100, lr=0.01, batch_size=128, optimizer=torch.optim.Rprop):
        super().__init__(__name__, "LSTM-AD")
        self.len_out = len_out

        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer_type = optimizer

        torch.manual_seed(0)

    def fit(self, X, _):
        self._build_model(X.shape[-1])

        X = np.expand_dims(X, axis=0)
        input_data = Variable(torch.from_numpy(X[:, :-self.len_out, :]), requires_grad=False)

        target_data = []
        for l in range(self.len_out - 1):
            target_data += [X[:, 1+l:-self.len_out+1+l, :]]
        target_data += [X[:, self.len_out:, :]]
        target_data = Variable(torch.from_numpy(np.stack(target_data, axis=3)), requires_grad=False)

        self._train_model(input_data, target_data)

    def predict(self, X):
        X = np.expand_dims(X, axis=0)
        input_data = Variable(torch.from_numpy(X[:, :-self.len_out, :]), requires_grad=False)
        target_data = []
        for l in range(self.len_out - 1):
            target_data += [X[:, 1+l:-self.len_out+1+l, :]]
        target_data += [X[:, self.len_out:, :]]
        target_data = Variable(torch.from_numpy(np.stack(target_data, axis=3)), requires_grad=False)

        predictions = self.model(input_data)

        errors = [predictions.data.numpy()[:, self.len_out-1:, :, 0]]
        for l in range(1, self.len_out):
            errors += [predictions.data.numpy()[:, self.len_out-1-l:-l, :, l]]
        errors = np.stack(errors, axis=3)
        errors = target_data.data.numpy()[:, self.len_out-1:, :, 0][..., np.newaxis] - errors

        SCALING_FACTOR = 1e10  # To compensate for lack of floating point precision
        # fit multivariate Gaussian on (validation set) error distribution (via maximum likelihood estimation)
        norm = errors.reshape(errors.shape[0] * errors.shape[1], X.shape[-1] * self.len_out)
        norm /= np.std(norm, axis=0)
        norm -= np.mean(norm, axis=0)
        norm *= SCALING_FACTOR
        mean = np.mean(norm, axis=0)
        cov = np.cov(norm.T)

        scores = -multivariate_normal.logpdf(norm, mean=mean, cov=cov) / np.log(SCALING_FACTOR)
        scores = np.pad(scores, (2 * self.len_out - 1, 0), 'constant', constant_values=np.nan)
        return scores

    def _build_model(self, d):
        self.model = LSTMSequence(d)
        self.model.double()

        self.loss = torch.nn.MSELoss()
        self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.lr)

    def _train_model(self, input_data, target_data):
        def closure():
            return self._train(input_data, target_data)

        for epoch in range(self.num_epochs):
            self.optimizer.step(closure)

    def _train(self, input_data, target_data):
        self.optimizer.zero_grad()
        output_data = self.model(input_data)
        loss_train = self.loss(output_data, target_data)
        loss_train.backward()
        return loss_train

    def binarize(self, score, threshold=None):
        threshold = self.threshold(score)
        score = np.where(np.isnan(score), threshold - 1, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2*np.nanstd(score)
