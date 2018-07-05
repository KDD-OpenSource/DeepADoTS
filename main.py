import os

import numpy as np
import pandas as pd

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMAD, LSTMED, LSTMAutoEncoder
from src.datasets import AirQuality, KDDCup, SyntheticDataGenerator
from src.evaluation.evaluator import Evaluator
from experiments import run_pollution_experiment, run_missing_experiment, run_extremes_experiment, \
    run_multivariate_experiment

# min number of runs = 2 for std operation
RUNS = 2 if os.environ.get("CIRCLECI", False) else 2


def main():
    run_pipeline()
    run_experiments()
    # test_stored_result()


def get_detectors():
    if os.environ.get("CIRCLECI", False):
        return [RecurrentEBM(num_epochs=2), Donut(num_epochs=5), LSTMAD(num_epochs=5), DAGMM(num_epochs=2),
                LSTMED(num_epochs=2), DAGMM(num_epochs=2, autoencoder_type=LSTMAutoEncoder)]
    else:
        return [RecurrentEBM(num_epochs=15), Donut(), LSTMAD(), LSTMED(num_epochs=40),
                DAGMM(sequence_length=1), DAGMM(sequence_length=15),
                DAGMM(sequence_length=15, autoencoder_type=LSTMAutoEncoder)]


def get_pipeline_datasets(seed):
    if os.environ.get("CIRCLECI", False):
        return [SyntheticDataGenerator.extreme_1(seed=42)]
    else:
        return [
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


def run_pipeline():
    datasets = get_pipeline_datasets(seed)
    detectors = get_detectors()

    # perform multiple pipeline runs for more robust end results
    # Set the random seed manually for reproducibility and more significant results
    # numpy expects a max. 32-bit unsigned integer
    seeds = np.random.randint(low=0, high=2 ** 32 - 1, size=RUNS, dtype="uint32")
    results = pd.DataFrame()
    evaluator = None

    for seed in seeds:
        evaluator = Evaluator(datasets, detectors, seed=seed)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

    evaluator.create_boxplots_per_algorithm(runs=RUNS, data=results)
    evaluator.create_boxplots_per_dataset(runs=RUNS, data=results)

    evaluator.gen_merged_tables(results)

    # set average results from multiple pipeline runs for evaluation
    avg_results = results.groupby(["dataset", "algorithm"], as_index=False).mean()
    evaluator.benchmark_results = avg_results
    evaluator.export_results('run-pipeline')

    evaluator.plot_threshold_comparison()
    evaluator.plot_single_heatmap()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    evaluator.create_bar_charts_per_dataset(runs=RUNS)
    evaluator.create_bar_charts_per_algorithm(runs=RUNS)


def test_stored_result():
    filename = 'run-pipeline-2018-07-03-165029'
    datasets = get_pipeline_datasets()
    detectors = get_detectors()
    evaluator = Evaluator(datasets, detectors)
    evaluator.import_results(filename)

    evaluator.print_tables()
    evaluator.plot_single_heatmap()


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


def run_experiments(outlier_type='extreme_1', output_dir=None, steps=5):
    output_dir = output_dir or os.path.join('reports/experiments', outlier_type)
    detectors = get_detectors()

    if os.environ.get("CIRCLECI", False):
        # Set the random seed manually for reproducibility and more significant results
        # numpy expects a max. 32-bit unsigned integer
        seeds = np.random.randint(low=0, high=2 ** 32 - 1, size=2, dtype="uint32")

        # min number of runs = 2 for std operation
        ev = run_extremes_experiment(detectors, seeds, runs=2, outlier_type=outlier_type,
                                     output_dir=os.path.join(output_dir,
                                                             'extremes'), steps=steps)
        ev.plot_single_heatmap()
    else:
        seeds = np.random.randint(low=0, high=2 ** 32 - 1, size=RUNS, dtype="uint32")
        announce_experiment('Pollution')
        ev_pol = run_pollution_experiment(detectors, seeds, RUNS, outlier_type,
                                          output_dir=os.path.join(output_dir, 'pollution'),
                                          steps=steps)

        announce_experiment('Missing Values')
        ev_mis = run_missing_experiment(detectors, seeds, RUNS, outlier_type,
                                        output_dir=os.path.join(output_dir, 'missing'),
                                        steps=steps)

        announce_experiment('Outlier height')
        ev_extr = run_extremes_experiment(detectors, seeds, RUNS, outlier_type,
                                          output_dir=os.path.join(output_dir, 'extremes'),
                                          steps=steps)

        announce_experiment('Multivariate Datasets')
        ev_mv = run_multivariate_experiment(detectors, seeds, RUNS, output_dir=os.path.join(output_dir, 'multivariate'))

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
