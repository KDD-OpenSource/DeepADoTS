import glob
import os

import numpy as np
import pandas as pd

from experiments import run_extremes_experiment, run_multivariate_experiment, run_multi_dim_multivariate_experiment, \
    announce_experiment, run_multivariate_polluted_experiment, run_different_window_sizes_evaluator
from src.algorithms import AutoEncoder, DAGMM, RecurrentEBM, LSTMAD, LSTMED
from src.datasets import KDDCup, RealPickledDataset
from src.evaluation import Evaluator

RUNS = 1


def main():
    run_experiments()


def detectors(seed):
    standard_epochs = 40
    dets = [AutoEncoder(num_epochs=standard_epochs, seed=seed),
            DAGMM(num_epochs=standard_epochs, seed=seed, lr=1e-4),
            DAGMM(num_epochs=standard_epochs, autoencoder_type=DAGMM.AutoEncoder.LSTM, seed=seed),
            LSTMAD(num_epochs=standard_epochs, seed=seed), LSTMED(num_epochs=standard_epochs, seed=seed),
            RecurrentEBM(num_epochs=standard_epochs, seed=seed)]

    return sorted(dets, key=lambda x: x.framework)


def run_experiments():
    # Set the seed manually for reproducibility.
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    output_dir = 'reports/experiments'
    evaluators = []
    outlier_height_steps = 10

    for outlier_type in ['extreme_1', 'shift_1', 'variance_1', 'trend_1']:
        announce_experiment('Outlier Height')
        ev_extr = run_extremes_experiment(
            detectors, seeds, RUNS, outlier_type, steps=outlier_height_steps,
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

    announce_experiment('Long-Term Experiments')
    ev_different_windows = run_different_window_sizes_evaluator(different_window_detectors, seeds, RUNS)
    evaluators.append(ev_different_windows)

    for ev in evaluators:
        ev.plot_single_heatmap()


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


def different_window_detectors(seed):
    standard_epochs = 40
    dets = [LSTMAD(num_epochs=standard_epochs)]
    for window_size in [13, 25, 50, 100]:
        dets.extend([LSTMED(name='LSTMED Window: ' + str(window_size), num_epochs=standard_epochs, seed=seed,
                            sequence_length=window_size), AutoEncoder(name='AE Window: ' + str(window_size),
                                                                      num_epochs=standard_epochs, seed=seed,
                                                                      sequence_length=window_size)])
    return dets


if __name__ == '__main__':
    main()
