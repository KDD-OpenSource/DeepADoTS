import numpy as np
import torch
from torch.autograd import Variable

from . import Algorithm

class LSTMSequence(torch.nn.Module):
    def __init__(self, d=2, len_in=1, len_out=10):
        super().__init__()
        self.d = d  # input and output feature dimensionality
        self.len_in = len_in
        self.len_out = len_out
        self.hidden_size1 = 32
        self.hidden_size2 = 32
        self.lstm1 = torch.nn.LSTMCell(d * len_in, self.hidden_size1)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_size1, self.hidden_size2)
        self.linear = torch.nn.Linear(self.hidden_size2, d * len_out)  # final linear layer as the output are real numbers

    def forward(self, input):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.hidden_size1).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.hidden_size1).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), self.hidden_size2).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), self.hidden_size2).double(), requires_grad=False)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):  # second dimension is the time dimension
            h_t, c_t = self.lstm1(input_t.squeeze(), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze()  # stack (n, d * len_out) outputs in time dimensionality (dim=1)
        
        return outputs.view(input.size(0), input.size(1), self.d, self.len_out)

class LSTMAD(Algorithm):
    """ LSTM-AD implementation using PyTorch.
    The interface of the class is sklearn-like.
    """

    def __init__(self, len_in=1, len_out=10, num_epochs=100, lr=0.01, optimizer=torch.optim.Rprop):
        self.len_in = len_in
        self.len_out = len_out

        self.num_epochs = num_epochs
        self.lr = lr

        self.optimizer_type = optimizer

        torch.manual_seed(0)

    def fit(self, X, _):
        self._build_model(X.shape[-1])

        input_data = Variable(torch.from_numpy(X[:, :-self.len_out, :]), requires_grad=False)

        target_data = []
        for l in range(self.len_out - 1):
            target_data += [X[:, 1+l:-self.len_out+1+l, :]]
        target_data += [X[:, self.len_out:, :]]
        target_data = Variable(torch.from_numpy(np.stack(target_train, axis=3)), requires_grad=False)

        self._train_model(input_data, target_data)

    def predict(self, X):
        input_data = Variable(torch.from_numpy(X[:, :-self.len_out, :]), requires_grad=False)
        target_data = []
        for l in range(self.len_out - 1):
            target_data += [X[:, 1+l:-self.len_out+1+l, :]]
        target_data += [X[:, self.len_out:, :]]

        predictions = self.model(input_data)

        errors = [pred_val.data.numpy()[:, self.len_out-1:, :, 0]]
        for l in range(1, self.len_out):
            errors += [predictions.data.numpy()[:, self.len_out-1-l:-l, :, l]]
        errors = np.stack(err_val, axis=3)
        errors = target_data.data.numpy()[:, self.len_out-1:, :, 0][..., np.newaxis] - errors

        # fit multivariate Gaussian on (validation set) error distribution (via maximum likelihood estimation)
        norm = errors.reshape(errors.shape[0] * errors.shape[1], d * self.len_out)
        mean = np.mean(norm, axis=0)
        sigma = (1.0 / len(norm)) * np.dot((norm - mean).T, norm_val - mean)

        scores = -multivariate_normal.logpdf(errors.reshape(errors.shape[0], errors.shape[1], d * self.len_out), 
                                             mean=mean, cov=sigma)
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
