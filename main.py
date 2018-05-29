import pickle
from typing import List

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.algorithms import DAGMM, LSTM_Enc_Dec
from src.datasets import KDD_Cup
from src.evaluation import get_accuracy_precision_recall_fscore


def main():
    # execute_dagmm()
    execute_lstm_enc_dec()


def execute_dagmm():
    dagmm = DAGMM()
    kdd_cup = KDD_Cup()
    (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print("DAGMM results: ", get_accuracy_precision_recall_fscore(y_test, pred))


def get_synthetic_data():
    with open("data/processed/synthetic", "rb") as f:
        (X_train, y_train, X_test, y_test) = pickle.load(f)
    return (X_train, y_train), (X_test, y_test)


def plot_thresholds(channel_scores: List[np.ndarray], y_test: pd.Series):
    steps = 40
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for channel_id, scores, ax in zip(range(4), channel_scores, axes.flat):
        maximum = scores.max()
        th = np.linspace(0, maximum, steps)
        anomalies_by_threshold = np.zeros(len(th))
        acc_by_threshold = np.zeros(len(th))
        prec_by_threshold = np.zeros(len(th))
        recall_by_threshold = np.zeros(len(th))
        f_score_by_threshold = np.zeros(len(th))
        for i in range(len(th)):
            anomaly = np.array(scores > th[i], dtype=int)
            anomalies_by_threshold[i] = anomaly.sum()
            acc_by_threshold[i], prec_by_threshold[i], recall_by_threshold[i], f_score_by_threshold[i] = get_accuracy_precision_recall_fscore(y_test, anomaly)
        ax.plot(th, anomalies_by_threshold / y_test.shape[0], label=r'anomalies ({} $\rightarrow$ 1)'.format(y_test.shape[0]))
        ax.plot(th, acc_by_threshold, label='accuracy')
        ax.plot(th, prec_by_threshold, label='precision')
        ax.plot(th, recall_by_threshold, label='recall')
        ax.plot(th, f_score_by_threshold, label='f_score')
        ax.set_title('Channel #{}'.format(channel_id + 1))
        ax.set_xlabel('Threshold')
        ax.legend()
    fig.tight_layout()
    fig.savefig('anomalies_by_threshold.png')
    fig.savefig('anomalies_by_threshold.pdf')


def execute_lstm_enc_dec():
    # (X_train, y_train), (X_test, y_test) = KDD_Cup().get_data_dagmm()
    (X_train, y_train), (X_test, y_test) = get_synthetic_data()

    lstm_enc_dec = LSTM_Enc_Dec(epochs=500, augment_train_data=True, data='lstm_enc_dec_not_augmented_2')
    lstm_enc_dec.fit(X_train, y_train)
    scores = lstm_enc_dec.predict_channel_scores(X_test)
    plot_thresholds(scores, y_test)
    binary_decisions = np.array(list(lstm_enc_dec.create_validation_set(scores)))
    pred = np.max(binary_decisions, axis=0)
    # pred = lstm_enc_dec.predict(X_test)

    print("LSTM-Enc_Dec results: ", get_accuracy_precision_recall_fscore(y_test, pred))


if __name__ == '__main__':
    main()
