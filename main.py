import pandas as pd
import numpy as np
from src.algorithms.donut import Donut
from src.datasets.air_quality import AirQuality


def main():
    # dagmm = DAGMM()
    # kdd_cup = KDD_Cup()
    # (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
    # dagmm.fit(X_train, y_train)
    # pred = dagmm.predict(X_test)
    # print(get_accuracy_precision_recall_fscore(y_test, pred))
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
    print(pred)


if __name__ == '__main__':
    main()
