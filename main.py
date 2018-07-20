import os

import numpy as np
import pandas as pd

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMAD, LSTMED, LSTMAutoEncoder
from src.datasets import AirQuality, KDDCup, SyntheticDataGenerator
from src.evaluation import Evaluator, Plotter
from experiments import run_pollution_experiment, run_missing_experiment, run_extremes_experiment, \
    run_multivariate_experiment, run_multi_dim_experiment, run_multi_dim_multivariate_experiment, announce_experiment

# Add this line if you want to test the pipeline & experiments
# os.environ['CIRCLECI'] = 'True'

# min number of runs = 2 for std operation
RUNS = 2 if os.environ.get('CIRCLECI', False) else 10


def main():
    run_pipeline()
    run_experiments()
    # run_final_missing_experiment(outlier_type='extreme_1', runs=100, only_load=False)


def detectors():
    if os.environ.get('CIRCLECI', False):
        dets = [RecurrentEBM(num_epochs=2), Donut(num_epochs=5), LSTMAD(num_epochs=5), DAGMM(num_epochs=2),
                LSTMED(num_epochs=2), DAGMM(num_epochs=2, autoencoder_type=LSTMAutoEncoder)]
    else:
        dets = [RecurrentEBM(num_epochs=15), Donut(), LSTMAD(), LSTMED(num_epochs=40),
                DAGMM(sequence_length=1), DAGMM(sequence_length=15),
                DAGMM(sequence_length=15, autoencoder_type=LSTMAutoEncoder)]
    return sorted(dets, key=lambda x: x.framework)


def pipeline_datasets(seed):
    if os.environ.get('CIRCLECI', False):
        return [SyntheticDataGenerator.extreme_1(seed)]
    else:
        return [
            SyntheticDataGenerator.extreme_1(seed),
            SyntheticDataGenerator.variance_1(seed),
            SyntheticDataGenerator.shift_1(seed),
            SyntheticDataGenerator.trend_1(seed),
            SyntheticDataGenerator.combined_1(seed),
            SyntheticDataGenerator.combined_4(seed),
        ]


def run_pipeline():
    # Perform multiple pipeline runs for more robust end results.
    # Set the seed manually for reproducibility.
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    results = pd.DataFrame()
    evaluator = None

    # Use same datasets for all CI runs (saves execution time)
    datasets = pipeline_datasets(42) if os.environ.get('CIRCLECI', False) else None

    for seed in seeds:
        datasets = datasets if datasets is not None else pipeline_datasets(seed)
        evaluator = Evaluator(datasets, detectors, seed=seed)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

        # Plots for each (det, ds, seed)
        evaluator.plot_roc_curves()
        evaluator.plot_scores()
        evaluator.plot_threshold_comparison()

    # Plots which need the whole data (not averaged)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)
    evaluator.gen_merged_tables(results)

    # Set average results from multiple pipeline runs for evaluation
    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.benchmark_results = avg_results
    evaluator.export_results('run-pipeline')

    # Plots using 'self.benchmark_results' (averaged)
    evaluator.plot_single_heatmap()
    evaluator.create_bar_charts(runs=RUNS, detectorwise=False)
    evaluator.create_bar_charts(runs=RUNS, detectorwise=True)


def run_experiments(steps=5):
    # Set the seed manually for reproducibility.
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)

    for outlier_type in ['extreme_1', 'shift_1', 'variance_1', 'trend_1']:
        output_dir = os.path.join('reports/experiments', outlier_type)

        announce_experiment('Outlier Height')
        ev_extr = run_extremes_experiment(
            detectors, seeds, RUNS, outlier_type, steps=10,
            output_dir=os.path.join(output_dir, 'extremes'))

        # CI: Keep the execution fast so stop after one experiment
        if os.environ.get('CIRCLECI', False):
            ev_extr.plot_single_heatmap()
            return

        announce_experiment('Pollution')
        ev_pol = run_pollution_experiment(
            detectors, seeds, RUNS, outlier_type, steps=steps,
            output_dir=os.path.join(output_dir, 'pollution'))

        announce_experiment('Missing Values')
        ev_mis = run_missing_experiment(
            detectors, seeds, RUNS, outlier_type, steps=steps,
            output_dir=os.path.join(output_dir, 'missing'))

        announce_experiment('High-dimensional normal outliers')
        ev_dim = run_multi_dim_experiment(
            detectors, outlier_type, RUNS, steps=20,
            output_dir=os.path.join(output_dir, 'multi_dim'))

    announce_experiment('Multivariate Datasets')
    ev_mv = run_multivariate_experiment(
        detectors, seeds, RUNS,
        output_dir=os.path.join(output_dir, 'multivariate'))

    announce_experiment('High-dimensional multivariate outliers')
    ev_mv_dim = run_multi_dim_multivariate_experiment(
        detectors, seeds, RUNS, steps=20,
        output_dir=os.path.join(output_dir, 'multi_dim_mv'))

    evaluators = [ev_pol, ev_mis, ev_extr, ev_mv, ev_dim, ev_mv_dim]
    for ev in evaluators:
        ev.plot_single_heatmap()


def run_final_missing_experiment(outlier_type='extreme_1', runs=25, output_dir=None, only_load=False, steps=5):
    output_dir = output_dir if output_dir is not None else os.path.join('reports/experiments', outlier_type)
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=runs, dtype=np.uint32)
    if not only_load:
        run_missing_experiment(
            detectors, seeds, RUNS, outlier_type, steps=steps,
            output_dir=output_dir, store_results=False)
    plotter = Plotter('reports', output_dir)
    plotter.plot_experiment('missing on extreme_1')


def evaluate_on_real_world_data_sets(seed):
    dagmm = DAGMM()
    dagmm.set_seed(seed)
    kdd_cup = KDDCup(seed)
    X_train, y_train, X_test, y_test = kdd_cup.data()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print(Evaluator.get_accuracy_precision_recall_fscore(y_test, pred))

    donut = Donut()
    donut.set_seed(seed)
    air_quality = AirQuality().data()
    X = air_quality.loc[:, [air_quality.columns[2], 'timestamps']]
    X['timestamps'] = X.index
    split_ratio = 0.8
    split_point = int(split_ratio * len(X))
    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = pd.Series(0, index=np.arange(len(X_train)))
    donut.fit(X_train, y_train)
    pred = donut.predict(X_test)
    print('Donut results: ', pred)


if __name__ == '__main__':
    main()
