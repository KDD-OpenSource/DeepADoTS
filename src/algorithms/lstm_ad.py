import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from torch.autograd import Variable
from tqdm import trange

from .algorithm_utils import Algorithm, PyTorchUtils


class LSTMAD(Algorithm, PyTorchUtils):
    """ LSTM-AD implementation using PyTorch.
    The interface of the class is sklearn-like.
    """

    def __init__(self, len_in=1, len_out=10, num_epochs=100, lr=1e-3, batch_size=1,
                 seed: int = None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, 'LSTM-AD', seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

        self.len_in = len_in
        self.len_out = len_out

        self.mean, self.cov = None, None

    def fit(self, X):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        self.batch_size = 1
        self._build_model(X.shape[-1], self.batch_size)

        self.model.train()
        split_point = int(0.75 * len(X))
        X_train = X.loc[:split_point, :]
        X_train_gaussian = X.loc[split_point:, :]

        input_data_train, target_data_train = self._input_and_target_data(X_train)
        self._train_model(input_data_train, target_data_train)

        self.model.eval()
        input_data_gaussian, target_data_gaussian = self._input_and_target_data_eval(X_train_gaussian)
        predictions_gaussian = self.model(input_data_gaussian)
        errors = self._calc_errors(predictions_gaussian, target_data_gaussian)

        # fit multivariate Gaussian on (validation set) error distribution (via maximum likelihood estimation)
        norm = errors.reshape(errors.shape[0] * errors.shape[1], X.shape[-1] * self.len_out)
        self.mean = np.mean(norm, axis=0)
        self.cov = np.cov(norm.T)

    def predict(self, X):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        self.model.eval()
        input_data, target_data = self._input_and_target_data_eval(X)

        predictions = self.model(input_data)
        errors, stacked_preds = self._calc_errors(predictions, target_data, return_stacked_predictions=True)

        if self.details:
            self.prediction_details.update({'predictions_mean': np.pad(
                stacked_preds.mean(axis=3).squeeze(0).T, ((0, 0), (self.len_in + self.len_out - 1, 0)),
                'constant', constant_values=np.nan)})
            self.prediction_details.update({'errors_mean': np.pad(
                errors.mean(axis=3).reshape(-1), (self.len_in + self.len_out - 1, 0),
                'constant', constant_values=np.nan)})

        norm = errors.reshape(errors.shape[0] * errors.shape[1], X.shape[-1] * self.len_out)
        scores = -multivariate_normal.logpdf(norm, mean=self.mean, cov=self.cov, allow_singular=True)
        scores = np.pad(scores, (self.len_in + self.len_out - 1, 0), 'constant', constant_values=np.nan)
        return scores

    def _input_and_target_data(self, X: pd.DataFrame):
        X = np.expand_dims(X, axis=0)
        input_data = self.to_var(torch.from_numpy(X[:, :-self.len_out, :]), requires_grad=False)
        target_data = []
        for l in range(self.len_out - 1):
            target_data += [X[:, 1 + l:-self.len_out + 1 + l, :]]
        target_data += [X[:, self.len_out:, :]]
        target_data = self.to_var(torch.from_numpy(np.stack(target_data, axis=3)), requires_grad=False)

        return input_data, target_data

    def _input_and_target_data_eval(self, X: pd.DataFrame):
        X = np.expand_dims(X, axis=0)
        input_data = self.to_var(torch.from_numpy(X), requires_grad=False)
        target_data = self.to_var(torch.from_numpy(X[:, self.len_in + self.len_out - 1:, :]), requires_grad=False)
        return input_data, target_data

    def _calc_errors(self, predictions, target_data, return_stacked_predictions=False):
        errors = [predictions.data.numpy()[:, self.len_out - 1:-self.len_in, :, 0]]
        for l in range(1, self.len_out):
            errors += [predictions.data.numpy()[:, self.len_out - 1 - l:-self.len_in - l, :, l]]
        errors = np.stack(errors, axis=3)
        stacked_predictions = errors
        errors = target_data.data.numpy()[..., np.newaxis] - errors
        return errors if return_stacked_predictions is False else (errors, stacked_predictions)

    def _build_model(self, d, batch_size):
        self.model = LSTMSequence(d, batch_size, len_in=self.len_in, len_out=self.len_out)
        self.to_device(self.model)
        self.model.double()

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _train_model(self, input_data, target_data):
        def closure():
            return self._train(input_data, target_data)

        for epoch in trange(self.num_epochs):
            self.optimizer.step(closure)

    def _train(self, input_data, target_data):
        self.optimizer.zero_grad()
        output_data = self.model(input_data)
        loss_train = self.loss(output_data, target_data)
        loss_train.backward()
        return loss_train


class LSTMSequence(torch.nn.Module):
    def __init__(self, d, batch_size: int, len_in=1, len_out=10):
        super().__init__()
        self.d = d  # input and output feature dimensionality
        self.batch_size = batch_size
        self.len_in = len_in
        self.len_out = len_out
        self.hidden_size1 = 32
        self.hidden_size2 = 32
        self.lstm1 = torch.nn.LSTMCell(d * len_in, self.hidden_size1)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_size1, self.hidden_size2)
        self.linear = torch.nn.Linear(self.hidden_size2, d * len_out)

        self.register_buffer('h_t', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('c_t', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('h_t2', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('c_t2', torch.zeros(self.batch_size, self.hidden_size1))

    def forward(self, input):
        outputs = []
        h_t = Variable(self.h_t.double(), requires_grad=False)
        c_t = Variable(self.c_t.double(), requires_grad=False)
        h_t2 = Variable(self.h_t2.double(), requires_grad=False)
        c_t2 = Variable(self.c_t2.double(), requires_grad=False)

        for input_t in input.chunk(input.size(1), dim=1):
            h_t, c_t = self.lstm1(input_t.squeeze(dim=1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze()  # stack (n, d * len_out) outputs in time dimensionality (dim=1)

        return outputs.view(input.size(0), input.size(1), self.d, self.len_out)
