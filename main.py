import os
import numpy as np
import pandas as pd

from src.datasets import AirQuality, KDDCup, SyntheticDataGenerator
from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMAD, LSTM_Enc_Dec
from src.evaluation.evaluator import Evaluator


def main():
    run_pipeline()


def run_pipeline():
    if os.environ.get("CIRCLECI", False):
        datasets = [SyntheticDataGenerator.extreme_1()]
        detectors = [RecurrentEBM(num_epochs=2), Donut(max_epoch=5), DAGMM(),
                     LSTM_Enc_Dec(epochs=2)]
    else:
        '''datasets = [
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
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(), Donut(), DAGMM(), LSTM_Enc_Dec(epochs=200)]'''
        datasets = [
            SyntheticDataGenerator.extreme_1(),
            SyntheticDataGenerator.variance_1()
        ]
        detectors = [RecurrentEBM(num_epochs=1), LSTM_Enc_Dec(epochs=1)]
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    evaluator.generate_latex()


def test_pollution():
    datasets = [
        SyntheticDataGenerator.extreme_1(),
        SyntheticDataGenerator.extreme_1_polluted(0.1),
        SyntheticDataGenerator.extreme_1_polluted(0.3),
        SyntheticDataGenerator.extreme_1_polluted(0.5),
        SyntheticDataGenerator.extreme_1_polluted(1)
    ]
    detectors = [RecurrentEBM(num_epochs=30), Donut(), DAGMM()]  # , LSTM_Enc_Dec(epochs=200)
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    evaluator.plot_auroc(title='Area under the curve for polluted data')


def test_missing():
    datasets = [
        SyntheticDataGenerator.extreme_1(),
        SyntheticDataGenerator.extreme_1_missing(0.1),
        SyntheticDataGenerator.extreme_1_missing(0.3),
        SyntheticDataGenerator.extreme_1_missing(0.5),
        SyntheticDataGenerator.extreme_1_missing(1)
    ]
    detectors = [RecurrentEBM(num_epochs=30), Donut(), DAGMM()]  # , LSTM_Enc_Dec(epochs=200)
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    evaluator.plot_auroc(title='Area under the curve for missing values')


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
