import os

import numpy as np
import pandas as pd

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMAD, LSTMED, LSTMAutoEncoder
from src.datasets import AirQuality, KDDCup, SyntheticDataGenerator
from src.evaluation.evaluator import Evaluator
from experiments import run_pollution_experiment, run_missing_experiment, run_extremes_experiment, \
                        run_multivariate_experiment

RUNS = 2


def main():
    run_pipeline()
    run_experiments()


def run_pipeline():
    if os.environ.get("CIRCLECI", False):
        datasets = [SyntheticDataGenerator.extreme_1()]
        detectors = [RecurrentEBM(num_epochs=2), LSTMAD(num_epochs=5), Donut(num_epochs=5), DAGMM(num_epochs=2),
                     LSTMED(num_epochs=2), DAGMM(num_epochs=2, autoencoder_type=LSTMAutoEncoder)]
    else:
        datasets = [
            SyntheticDataGenerator.extreme_1(),
            SyntheticDataGenerator.variance_1(),
            SyntheticDataGenerator.shift_1(),
            SyntheticDataGenerator.trend_1(),
            SyntheticDataGenerator.combined_1(),
            SyntheticDataGenerator.combined_4(),
        ]
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(), Donut(), LSTMED(num_epochs=40),
                     DAGMM(sequence_length=1), DAGMM(sequence_length=15),
                     DAGMM(sequence_length=1, autoencoder_type=LSTMAutoEncoder),
                     DAGMM(sequence_length=15, autoencoder_type=LSTMAutoEncoder)]

    evaluator = Evaluator(datasets, detectors)
    # perform multiple pipeline runs for more significant end results
    results = pd.DataFrame()
    for _ in range(RUNS):
        evaluator.evaluate()
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

    evaluator.create_boxplots_per_algorithm(runs=RUNS, data=results)
    evaluator.create_boxplots_per_dataset(runs=RUNS, data=results)

    # average results from multiple pipeline runs
    averaged_results = results.groupby(["dataset", "algorithm"], as_index=False).mean()
    evaluator.benchmark_results = averaged_results

    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_single_heatmap()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    evaluator.create_bar_charts_per_dataset(runs=RUNS)
    evaluator.create_bar_charts_per_algorithm(runs=RUNS)
    evaluator.generate_latex()


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


def run_experiments(outlier_type='extreme_1', output_dir=None, steps=5):
    output_dir = output_dir or os.path.join('reports/experiments', outlier_type)
    if os.environ.get("CIRCLECI", False):
        detectors = [RecurrentEBM(num_epochs=2), LSTMAD(num_epochs=5), Donut(num_epochs=5),
                     LSTMED(num_epochs=2), DAGMM(num_epochs=2),
                     DAGMM(num_epochs=2, autoencoder_type=LSTMAutoEncoder)]
        ev = run_extremes_experiment(detectors, outlier_type, output_dir=os.path.join(output_dir, 'extremes'),
                                     steps=1)
        ev.plot_single_heatmap()
    else:
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(), Donut(), LSTMED(num_epochs=40),
                     DAGMM(sequence_length=1),
                     DAGMM(sequence_length=15),
                     DAGMM(sequence_length=1, autoencoder_type=LSTMAutoEncoder),
                     DAGMM(sequence_length=15, autoencoder_type=LSTMAutoEncoder)]

        announce_experiment('Pollution')
        ev_pol = run_pollution_experiment(detectors, outlier_type, output_dir=os.path.join(output_dir, 'pollution'),
                                          steps=steps)

        announce_experiment('Missing Values')
        ev_mis = run_missing_experiment(detectors, outlier_type, output_dir=os.path.join(output_dir, 'missing'),
                                        steps=steps)

        announce_experiment('Outlier height')
        ev_extr = run_extremes_experiment(detectors, outlier_type, output_dir=os.path.join(output_dir, 'extremes'),
                                          steps=steps)

        announce_experiment('Multivariate Datasets')
        ev_mv = run_multivariate_experiment(detectors, output_dir=os.path.join(output_dir, 'multivariate'))

        evaluators = [ev_pol, ev_mis, ev_extr, ev_mv]
        Evaluator.plot_heatmap(evaluators)


def announce_experiment(title: str, dashes: int = 70):
    print(f'\n###{"-"*dashes}###')
    message = f'Experiment: {title}'
    before = (dashes - len(message)) // 2
    after = dashes - len(message) - before
    print(f'###{"-"*before}{message}{"-"*after}###')
    print(f'###{"-"*dashes}###\n')


if __name__ == '__main__':
    main()
