import logging
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTM_Enc_Dec, LSTMAD
from src.datasets.air_quality import AirQuality
from src.datasets.kdd_cup import KDDCup
from src.datasets.synthetic_data_generator import SyntheticDataGenerator
from src.evaluation.evaluator import Evaluator


def evaluate_on_real_world_data_sets():
    dagmm = DAGMM()
    kdd_cup = KDDCup()
    (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
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
    print(pred)


def main():
    if os.environ.get("CIRCLECI", False):
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.INFO)
        datasets = [SyntheticDataGenerator.extreme_1()]
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(num_epochs=10), Donut(max_epoch=5), DAGMM()]
        # LSTM_Enc_Dec(epochs=10)] TODO: Issue #48
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
        detectors = [ DAGMM()]
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    df = evaluator.benchmarks()
    for ds in df['dataset'].unique():
        print("Dataset: " + ds)
        print_order = ["algorithm", "accuracy", "precision", "recall", "F1-score", "F0.1-score"]
        print(tabulate(df[df['dataset'] == ds][print_order], headers='keys', tablefmt='psql'))
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
