import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """AutoEncoder class, forward needs to return (decoded, encoded)"""


class NNAutoEncoder(AutoEncoder):

    def __init__(self, n_features=118, sequence_length=1, hidden_size=1):
        super().__init__()

        n_features = n_features * sequence_length

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

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        enc = self._encoder(x)
        dec = self._decoder(enc)

        return dec, enc


class LSTMAutoEncoder(AutoEncoder):
    """Autoencoder with Recurrent module. Inspired by LSTM-EncDec"""

    def __init__(self, n_features: int, sequence_length: int, hidden_size: int = 1, n_layers: tuple = (3, 3),
                 use_bias: tuple = (True, True), dropout: tuple = (0.3, 0.3)):
        super().__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0]).cuda()
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1]).cuda()
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features).cuda()

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers[0], batch_size, self.hidden_size).cuda(),
                torch.zeros(self.n_layers[0], batch_size, self.hidden_size).cuda())

    def forward(self, ts_batch):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(ts_batch.shape[0])  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = (enc_hidden[0], torch.zeros(self.n_layers[1], batch_size, self.hidden_size).cuda())

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = torch.zeros(ts_batch.size()).cuda()
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return output.squeeze(2), enc_hidden[0][-1].view(batch_size, self.hidden_size)
