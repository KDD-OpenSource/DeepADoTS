import os
import numpy as np
import pandas as pd

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMEncDec, LSTMAD, EnsembleLSTMEncDec
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
        datasets = [SyntheticDataGenerator.extreme_1()]
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(num_epochs=10), Donut(max_epoch=5), DAGMM(),
                     LSTM_Enc_Dec(epochs=10)]
    else:
        datasets = [
            SyntheticDataGenerator.extreme_1(),
        ]
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(), Donut(), DAGMM(), LSTM_Enc_Dec(epochs=200), Ensem]
        detectors = [LSTM_Enc_Dec(epochs=10)]
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    df = evaluator.benchmarks()
    print('Evaluated benchmarks: ', df)
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
