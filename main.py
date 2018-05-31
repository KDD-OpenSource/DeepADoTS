import pickle

import pandas as pd
import numpy as np

from src.datasets import AirQuality, SyntheticDataset, KDDCup
from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMAD
from src.evaluation.evaluator import Evaluator


def main():
    # execute_dagmm()
    # execute_donut()
    execute_pipeline()


def execute_pipeline():
    datasets = [SyntheticDataset(name="Synthetic Extreme Outliers", file_name="synthetic"), SyntheticDataset(name="Synthetic Extreme Outliers 2", file_name="synthetic2")]
    detectors = [RecurrentEBM(num_epochs=15), LSTMAD()]
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    df = evaluator.benchmarks()
    print('Evaluated benchmarks: ', df)
    evaluator.plot_threshold_comparison()
    # evaluator.plot_scores()


def execute_donut():
    donut = Donut()
    air_quality = AirQuality().get_data()
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


def execute_dagmm():
    dagmm = DAGMM()
    kdd_cup = KDDCup()
    (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print("DAGMM results: ", Evaluator.get_accuracy_precision_recall_fscore(y_test, pred))


if __name__ == '__main__':
    main()
