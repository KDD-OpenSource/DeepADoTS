import time
import logging
from pathlib import Path

import argparse
import torch
import torch.nn as nn
from .preprocess_data import *
from .model import RNNPredictor
from torch import optim
from matplotlib import pyplot as plt
from .anomalyDetector import fit_norm_distribution_param

REPORT_FIGURES_DIR = 'reports/figures'

parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset')
# We use the network name (instead of 'ecg') because it stores its trained model with this name
parser.add_argument('--data', type=str, default='lstm_enc_dec',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
                    help='filename of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--augment_train_data', type=bool, default=True,
                    help='augment train data')
parser.add_argument('--emsize', type=int, default=32,
                    help='size of rnn input features')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--res_connection', action='store_true',
                    help='residual connection')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=10,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=3,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval_batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                    help='teacher forcing ratio (deprecated)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights (deprecated)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cpu',
                    help='cuda or cpu')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval')
parser.add_argument('--resume', '-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained', '-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def get_args():
    return args

def set_args(**kwargs):
    global args
    parser.set_defaults(**kwargs)
    args = parser.parse_args()


###############################################################################
# Training code
###############################################################################
def get_batch(args, source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]  # [ seq_len * batch_size * feature_size ]
    target = source[i + 1:i + 1 + seq_len]  # [ (seq_len x batch_size x feature_size) ]
    return data, target

def train(args, model, train_dataset, epoch, optimizer, criterion):
    with torch.enable_grad():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        hidden = model.init_hidden(args.batch_size)
        for batch, i in enumerate(range(0, train_dataset.size(0) - 1, args.bptt)):
            inputSeq, targetSeq = get_batch(args, train_dataset, i)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = model.repackage_hidden(hidden)
            hidden_ = model.repackage_hidden(hidden)
            optimizer.zero_grad()

            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)
            outVals = []
            hids1 = []
            for i in range(inputSeq.size(0)):
                outVal, hidden_, hid = model.forward(outVal, hidden_, return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals, dim=0)
            hids1 = torch.cat(hids1, dim=0)

            loss1 = criterion(outSeq1, targetSeq)

            '''Loss2: Teacher forcing loss'''
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2, targetSeq)

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1, hids2.detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1 + loss2 + loss3
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.4f} | '
                      'loss {:5.2f} '.format(
                    epoch, batch, len(train_dataset) // args.bptt,
                                  elapsed * 1000 / args.log_interval, cur_loss))
                total_loss = 0
                start_time = time.time()


def evaluate(args, model, test_dataset, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        total_loss = 0
        hidden = model.init_hidden(args.eval_batch_size)
        nbatch = 1
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):
            inputSeq, targetSeq = get_batch(args, test_dataset, i)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]
            hidden_ = model.repackage_hidden(hidden)
            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)
            outVals = []
            hids1 = []
            for i in range(inputSeq.size(0)):
                outVal, hidden_, hid = model.forward(outVal, hidden_, return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals, dim=0)
            hids1 = torch.cat(hids1, dim=0)
            loss1 = criterion(outSeq1, targetSeq)

            '''Loss2: Teacher forcing loss'''
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2, targetSeq)

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.view(args.batch_size, -1), hids2.view(args.batch_size, -1).detach())
            loss3 = criterion(hids1, hids2.detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1 + loss2 + loss3

            total_loss += loss.item()

    return total_loss / (nbatch + 1)
