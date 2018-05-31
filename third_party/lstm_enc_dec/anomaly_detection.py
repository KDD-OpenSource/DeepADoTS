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
from .train_predictor import get_args as get_train_args

REPORT_PICKLES_DIR = 'reports/data'
REPORT_FIGURES_DIR = 'reports/figures'


def calc_anomalies(TimeseriesData, train_dataset, test_dataset):
    train_args = get_train_args()
    parser = argparse.ArgumentParser(description='PyTorch LSTM-Enc-Dec Anomaly Detection Model')
    parser.add_argument('--prediction_window_size', type=int, default=train_args.prediction_window_size,
                        help='prediction_window_size')
    parser.add_argument('--data', type=str, default=train_args.data,
                        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
                        help='filename of the dataset')
    parser.add_argument('--compensate', action='store_true',
                        help='compensate anomaly score using anomaly score esimation')

    args_ = parser.parse_args()
    logging.info('-' * 89)
    logging.info("=> loading checkpoint ")
    checkpoint = torch.load(str(Path('models', args_.data, 'checkpoint', args_.filename).with_suffix('.pth')))
    args = checkpoint['args']
    args.prediction_window_size = args_.prediction_window_size
    args.compensate = args_.compensate
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
    model = RNNPredictor(rnn_type=args.model,
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
                logging.info('=> loading pre-calculated mean and covariance')
                mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
            else:
                logging.info('=> calculating mean and covariance')
                mean, cov = fit_norm_distribution_param(args, model, train_dataset, channel_idx=channel_idx)

            ''' 2. Train anomaly score predictor using support vector regression (SVR). (Optional) '''
            # An anomaly score predictor is trained
            # given hidden layer output and the corresponding anomaly score on train dataset.
            # Predicted anomaly scores on test dataset can be used for the baseline of the adaptive threshold.
            if args.compensate:
                logging.info('=> training an SVR as anomaly score predictor')
                train_score, _, _, hiddens, _ = anomalyScore(args, model, train_dataset, mean, cov,
                                                             channel_idx=channel_idx)
                score_predictor = GridSearchCV(SVR(), cv=5,
                                               param_grid={"C": [1e0, 1e1, 1e2], "gamma": np.logspace(-1, 1, 3)})
                score_predictor.fit(torch.cat(hiddens, dim=0).numpy(), train_score.cpu().numpy())
            else:
                score_predictor = None

            ''' 3. Calculate anomaly scores'''
            # Anomaly scores are calculated on the test dataset
            # given the mean and the covariance calculated on the train dataset
            logging.info('=> calculating anomaly scores')
            score, _, _, _, predicted_score = anomalyScore(
                args, model, test_dataset, mean, cov,
                score_predictor=score_predictor, channel_idx=channel_idx
            )
            score = score.cpu()
            scores.append(score)
            predicted_scores.append(predicted_score)

    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')

    logging.info('=> saving the results as pickle extensions')
    save_dir = Path(REPORT_PICKLES_DIR, args.data, args.filename).with_suffix('')
    save_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(scores, open(str(save_dir.joinpath('score.pkl')), 'wb'))
    pickle.dump(predicted_scores, open(str(save_dir.joinpath('predicted_scores.pkl')), 'wb'))
    logging.info('-' * 89)

    return scores, predicted_scores
