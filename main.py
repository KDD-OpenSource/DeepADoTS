import sys
import glob
import os

import numpy as np
import pandas as pd

from experiments import run_pollution_experiment, run_missing_experiment, run_extremes_experiment, \
    run_multivariate_experiment, run_multi_dim_experiment, run_multi_dim_multivariate_experiment, announce_experiment, \
    run_multivariate_polluted_experiment
from src.algorithms import AutoEncoder, DAGMM, Donut, RecurrentEBM, LSTMAD, LSTMED
from src.datasets import KDDCup, SyntheticDataGenerator, RealPickledDataset
from src.evaluation import Evaluator, Plotter

# Add this line if you want to test the pipeline & experiments
# os.environ['CIRCLECI'] = 'True'

RUNS = 2 if os.environ.get('CIRCLECI', False) else 15


def main():
    run_experiments()


def detectors(seed):
    if os.environ.get('CIRCLECI', False):
        dets = [AutoEncoder(num_epochs=1, seed=seed), DAGMM(num_epochs=1, seed=seed),
                DAGMM(num_epochs=1, autoencoder_type=DAGMM.AutoEncoder.LSTM, seed=seed),
                Donut(num_epochs=1, seed=seed), LSTMAD(num_epochs=1, seed=seed), LSTMED(num_epochs=1, seed=seed),
                RecurrentEBM(num_epochs=1, seed=seed)]
    else:
        dets = [AutoEncoder(num_epochs=40, seed=seed), DAGMM(sequence_length=15, seed=seed),
                DAGMM(sequence_length=15, autoencoder_type=DAGMM.AutoEncoder.LSTM, seed=seed),
                Donut(seed=seed), LSTMAD(seed=seed), LSTMED(num_epochs=40, seed=seed),
                RecurrentEBM(num_epochs=15, seed=seed)]
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
    output_dir = 'reports/experiments'
    evaluators = []

    for outlier_type in ['extreme_1', 'shift_1', 'variance_1', 'trend_1']:
        announce_experiment('Outlier Height')
        ev_extr = run_extremes_experiment(
            detectors, seeds, RUNS, outlier_type, steps=10,
            output_dir=os.path.join(output_dir, outlier_type, 'intensity'))
        evaluators.append(ev_extr)

    announce_experiment('Multivariate Datasets')
    ev_mv = run_multivariate_experiment(
        detectors, seeds, RUNS,
        output_dir=os.path.join(output_dir, 'multivariate'))
    evaluators.append(ev_mv)

    for mv_anomaly in ['doubled', 'inversed', 'shrinked', 'delayed', 'xor', 'delayed_missing']:
        announce_experiment(f'Multivariate Polluted {mv_anomaly} Datasets')
        ev_mv = run_multivariate_polluted_experiment(
            detectors, seeds, RUNS, mv_anomaly,
            output_dir=os.path.join(output_dir, 'mv_polluted'))
        evaluators.append(ev_mv)

        announce_experiment(f'High-dimensional multivariate {mv_anomaly} outliers')
        ev_mv_dim = run_multi_dim_multivariate_experiment(
            detectors, seeds, RUNS, mv_anomaly, steps=20,
            output_dir=os.path.join(output_dir, 'multi_dim_mv'))
        evaluators.append(ev_mv_dim)

    # TODO: Window-length experiment

    for ev in evaluators:
        ev.plot_single_heatmap()


def run_final_missing_experiment(outlier_type='extreme_1', runs=25, steps=5):
    only_load = len(sys.argv) > 1 and sys.argv[1] == 'load'
    output_dir = os.path.join('reports/experiments', outlier_type)
    if len(sys.argv) > 2:
        output_dir = os.path.join('reports', sys.argv[2], outlier_type)
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=runs, dtype=np.uint32)
    if not only_load:
        run_missing_experiment(
            detectors, seeds, RUNS, outlier_type, steps=steps,
            output_dir=output_dir, store_results=False)
    plotter = Plotter('reports', output_dir)
    plotter.barplots(f'missing on {outlier_type}')


def evaluate_real_datasets():
    REAL_DATASET_GROUP_PATH = 'data/raw/'
    real_dataset_groups = glob.glob(REAL_DATASET_GROUP_PATH + '*')
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    results = pd.DataFrame()
    datasets = [KDDCup(seed=1)]
    for real_dataset_group in real_dataset_groups:
        for data_set_path in glob.glob(real_dataset_group + '/labeled/train/*'):
            data_set_name = data_set_path.split('/')[-1].replace('.pkl', '')
            dataset = RealPickledDataset(data_set_name, data_set_path)
            datasets.append(dataset)

    for seed in seeds:
        datasets[0] = KDDCup(seed)
        evaluator = Evaluator(datasets, detectors, seed=seed)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()
        results = results.append(result, ignore_index=True)

    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results('run_real_datasets')
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=RUNS, data=results, detectorwise=True)


if __name__ == '__main__':
    main()
