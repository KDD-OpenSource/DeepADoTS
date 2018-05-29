# import pandas as pd
# import numpy as np
# from src.algorithms.donut import Donut
# from src.datasets.air_quality import AirQuality
from src.datasets.synthetic_data_generator import SyntheticData
from src.algorithms import RecurrentEBM
from src.algorithms import LSTMAD
from src.evaluation.evaluator import Evaluator


def main():
    # dagmm = DAGMM()
    # kdd_cup = KDD_Cup()
    # (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
    # dagmm.fit(X_train, y_train)
    # pred = dagmm.predict(X_test)
    # print(get_accuracy_precision_recall_fscore(y_test, pred))

    # donut = Donut()
    # air_quality = AirQuality().get_data()
    # X = air_quality.loc[:, [air_quality.columns[2], "timestamps"]]
    # X["timestamps"] = X.index
    # split_ratio = 0.8
    # split_point = int(split_ratio * len(X))
    # X_train = X[:split_point]
    # X_test = X[split_point:]
    # y_train = pd.Series(0, index=np.arange(len(X_train)))
    # donut.fit(X_train, y_train)
    # pred = donut.predict(X_test)
    # print(pred)

    datasets = [SyntheticData("Synthetic Extreme Outliers", ".")]
    detectors = [RecurrentEBM(num_epochs=15), LSTMAD()]
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    df = evaluator.benchmarks()
    print(df)
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
