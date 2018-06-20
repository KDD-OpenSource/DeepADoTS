import os

import numpy as np
import pandas as pd

from src.algorithms import DAGMM, Donut, LSTM_Enc_Dec, LSTMAD, LSTMAutoEncoder, LSTMED, RecurrentEBM
from src.datasets import AirQuality, KDDCup, SyntheticDataGenerator
from src.evaluation.evaluator import Evaluator
# from src.evaluation.experiments import run_experiments


def main():
    run_pipeline()
    # run_experiments()


def run_pipeline():
    if os.environ.get("CIRCLECI", False):
        datasets = [SyntheticDataGenerator.extreme_1()]
        detectors = [DAGMM(sequence_length=15, autoencoder_type=LSTMAutoEncoder, lr=1e-3), Donut(max_epoch=5),
                     LSTM_Enc_Dec(epochs=2), LSTMAD(num_epochs=5), LSTMED(epochs=2), RecurrentEBM(num_epochs=2),
                     DAGMM(sequence_length=1), DAGMM(sequence_length=15)]
    else:
        datasets = [
            SyntheticDataGenerator.extreme_1(),
            SyntheticDataGenerator.variance_1(),
            SyntheticDataGenerator.shift_1(),
            SyntheticDataGenerator.trend_1(),
            SyntheticDataGenerator.combined_1(),
            SyntheticDataGenerator.combined_4(),
            SyntheticDataGenerator.variance_1_missing(0.1),
            SyntheticDataGenerator.variance_1_missing(0.3),
            SyntheticDataGenerator.variance_1_missing(0.5),
            SyntheticDataGenerator.variance_1_missing(0.8),
            SyntheticDataGenerator.extreme_1_polluted(0.1),
            SyntheticDataGenerator.extreme_1_polluted(0.3),
            SyntheticDataGenerator.extreme_1_polluted(0.5),
            SyntheticDataGenerator.extreme_1_polluted(1)
        ]
        detectors = [RecurrentEBM(num_epochs=15), LSTMED(hidden_size=4, epochs=40), LSTMAD(num_epochs=5),
                     Donut(), DAGMM(sequence_length=1), DAGMM(sequence_length=15),
                     DAGMM(sequence_length=15, autoencoder_type=LSTMAutoEncoder, lr=1e-3)]
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()


def evaluate_on_real_world_data_sets():
    dagmm = DAGMM()
    kdd_cup = KDDCup()
    X_train, y_train, X_test, y_test = kdd_cup.data()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print(Evaluator.get_accuracy_precision_recall_fscore(y_test, pred))

    donut = Donut()
    air_quality = AirQuality().data()
    X = air_quality.loc[:, [air_quality.columns[2], "timestamps"]]
    X["timestamps"] = X.index
    split_ratio = 0.8
    split_point = int(split_ratio * len(X))
    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = pd.Series(0, index=np.arange(len(X_train)))
    donut.fit(X_train, y_train)
    pred = donut.predict(X_test)
    print("Donut results: ", pred)


if __name__ == '__main__':
    main()
