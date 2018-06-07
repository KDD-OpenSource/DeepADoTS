import logging
from pathlib import Path

import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from .anomalyDetector import anomalyScore
from .anomalyDetector import fit_norm_distribution_param
from .model import RNNPredictor
from .preprocess_data import *

REPORT_PICKLES_DIR = 'reports/data'
REPORT_FIGURES_DIR = 'reports/figures'


def calc_anomalies(TimeseriesData, train_dataset, test_dataset, device_type, data,
                   filename, compensate):
    logging.debug('-' * 89)
    logging.debug("=> loading checkpoint ")
    checkpoint = torch.load(str(Path('models', data, 'checkpoint', filename).with_suffix('.pth')))
    # Contains seed, model_type, emsize, nhid, nlayers, res_connection, prediction_window_size, device
    args = checkpoint['args']
    logging.info("=> loaded checkpoint")

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    TimeseriesData = TimeseriesData
    train_dataset = train_dataset
    test_dataset = test_dataset

    ###############################################################################
    # Build the model
    ###############################################################################
    nfeatures = TimeseriesData.trainData.size(-1)
    model = RNNPredictor(rnn_type=args.model_type,
                         enc_inp_size=nfeatures,
                         rnn_inp_size=args.emsize,
                         rnn_hid_size=args.nhid,
                         dec_out_size=nfeatures,
                         nlayers=args.nlayers,
                         res_connection=args.res_connection).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    # del checkpoint

    scores, predicted_scores = list(), list()
    try:
        # For each channel in the dataset
        for channel_idx in range(nfeatures):
            ''' 1. Load mean and covariance if they are pre-calculated, if not calculate them. '''
            # Mean and covariance are calculated on train dataset.
            if 'means' in checkpoint.keys() and 'covs' in checkpoint.keys():
                logging.debug('=> loading pre-calculated mean and covariance')
                mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
            else:
                logging.debug('=> calculating mean and covariance')
                mean, cov = fit_norm_distribution_param(
                    model, train_dataset, args.prediction_window_size, args.device,
                    channel_idx=channel_idx)

            ''' 2. Train anomaly score predictor using support vector regression (SVR). (Optional) '''
            # An anomaly score predictor is trained
            # given hidden layer output and the corresponding anomaly score on train dataset.
            # Predicted anomaly scores on test dataset can be used for the baseline of the adaptive threshold.
            if compensate:
                logging.debug('=> training an SVR as anomaly score predictor')
                train_score, _, _, hiddens, _ = anomalyScore(
                    model, train_dataset, mean, cov, args.prediction_window_size, args.device,
                    channel_idx=channel_idx)
                score_predictor = GridSearchCV(SVR(), cv=5,
                                               param_grid={"C": [1e0, 1e1, 1e2], "gamma": np.logspace(-1, 1, 3)})
                score_predictor.fit(torch.cat(hiddens, dim=0).numpy(), train_score.cpu().numpy())
            else:
                score_predictor = None

            ''' 3. Calculate anomaly scores'''
            # Anomaly scores are calculated on the test dataset
            # given the mean and the covariance calculated on the train dataset
            logging.debug('=> calculating anomaly scores')
            score, _, _, _, predicted_score = anomalyScore(
                model, test_dataset, mean, cov,
                args.prediction_window_size, args.device,
                score_predictor=score_predictor, channel_idx=channel_idx,
            )
            score = score.cpu()
            scores.append(score)
            predicted_scores.append(predicted_score)

    except KeyboardInterrupt:
        logging.warn('-' * 89)
        logging.warn('Exiting from training early')

    logging.debug('=> saving the results as pickle extensions')
    save_dir = Path(REPORT_PICKLES_DIR, data, filename).with_suffix('')
    save_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(scores, open(str(save_dir.joinpath('score.pkl')), 'wb'))
    pickle.dump(predicted_scores, open(str(save_dir.joinpath('predicted_scores.pkl')), 'wb'))
    logging.debug('-' * 89)

    return scores, predicted_scores
