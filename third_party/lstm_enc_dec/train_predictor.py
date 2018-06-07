import logging

import time

from .preprocess_data import *

REPORT_FIGURES_DIR = 'reports/figures'


###############################################################################
# Training code
###############################################################################
def get_batch(seq_length, source, i):
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i + seq_len]  # [ seq_len * batch_size * feature_size ]
    target = source[i + 1:i + 1 + seq_len]  # [ (seq_len x batch_size x feature_size) ]
    return data, target


def train(model, train_dataset, epoch, optimizer, criterion,
          batch_size, seq_length, log_interval, gradient_clip):
    with torch.enable_grad():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        hidden = model.init_hidden(batch_size)
        for batch, i in enumerate(range(0, train_dataset.size(0) - 1, seq_length)):
            inputSeq, targetSeq = get_batch(seq_length, train_dataset, i)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            total_loss += loss.item()

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                logging.debug('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.4f} | '
                             'loss {:5.2f} '.format(
                    epoch, batch, len(train_dataset) // seq_length,
                                  elapsed * 1000 / log_interval, cur_loss))
                total_loss = 0
                start_time = time.time()


def evaluate(model, test_dataset, criterion, batch_size, eval_batch_size, seq_length):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        total_loss = 0
        hidden = model.init_hidden(eval_batch_size)
        nbatch = 1
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, seq_length)):
            inputSeq, targetSeq = get_batch(seq_length, test_dataset, i)
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
            loss3 = criterion(hids1.view(batch_size, -1), hids2.view(batch_size, -1).detach())
            loss3 = criterion(hids1, hids2.detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1 + loss2 + loss3

            total_loss += loss.item()

    return total_loss / (nbatch + 1)
