import os
import sys

import numpy as np
import pandas as pd

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMAD, LSTM_Enc_Dec
from src.datasets import AirQuality, KDDCup, SyntheticDataGenerator
from src.evaluation.evaluator import Evaluator


# from src.evaluation.experiments import run_experiments

RUNS = 2


def main():
    run_pipeline()
    # run_experiments()


def run_pipeline():
    datasets = None
    if os.environ.get("CIRCLECI", False):
        datasets = [SyntheticDataGenerator.extreme_1(seed=42)]
        detectors = [RecurrentEBM(num_epochs=2), LSTMAD(num_epochs=5), Donut(num_epochs=5), DAGMM(),
                     LSTM_Enc_Dec(num_epochs=2)]
    else:
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(), Donut(), DAGMM(), LSTM_Enc_Dec(num_epochs=15)]

    # perform multiple pipeline runs for more significant end results
    # Set the random seed manually for reproducibility and more significant results
    # numpy expects a max. 32-bit unsigned integer
    seeds = np.random.randint(low=0, high=2**32 - 1, size=RUNS)
    results = pd.DataFrame()
    evaluator = None

    for seed in seeds:
        evaluator = Evaluator(datasets if datasets else get_datasets(seed), detectors)
        evaluator.evaluate(seed)
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

    evaluator.create_boxplots_per_algorithm(runs=RUNS, data=results)
    evaluator.create_boxplots_per_dataset(runs=RUNS, data=results)

    # average results from multiple pipeline runs
    averaged_results = results.groupby(["dataset", "algorithm"], as_index=False).mean()
    evaluator.benchmark_results = averaged_results

    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    evaluator.create_bar_charts_per_dataset(runs=RUNS)
    evaluator.create_bar_charts_per_algorithm(runs=RUNS)

def evaluate_on_real_world_data_sets(seed):
    dagmm = DAGMM()
    dagmm.set_seed(seed)
    # numpy expects a 32-bit unsigned integer
    kdd_cup = KDDCup(seed)
    X_train, y_train, X_test, y_test = kdd_cup.data()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print(Evaluator.get_accuracy_precision_recall_fscore(y_test, pred))

    donut = Donut()
    donut.set_seed(seed)
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

def get_datasets(seed):
    datasets = [
        SyntheticDataGenerator.extreme_1(seed),
        SyntheticDataGenerator.variance_1(seed),
        SyntheticDataGenerator.shift_1(seed),
        SyntheticDataGenerator.trend_1(seed),
        SyntheticDataGenerator.combined_1(seed),
        SyntheticDataGenerator.combined_4(seed),
        SyntheticDataGenerator.variance_1_missing(seed, 0.1),
        SyntheticDataGenerator.variance_1_missing(seed, 0.3),
        SyntheticDataGenerator.variance_1_missing(seed, 0.5),
        SyntheticDataGenerator.variance_1_missing(seed, 0.8),
        SyntheticDataGenerator.extreme_1_polluted(seed, 0.1),
        SyntheticDataGenerator.extreme_1_polluted(seed, 0.3),
        SyntheticDataGenerator.extreme_1_polluted(seed, 0.5),
        SyntheticDataGenerator.extreme_1_polluted(seed, 0.9)
    ]
    return datasets


if __name__ == '__main__':
    main()
