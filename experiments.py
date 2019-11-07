import numpy as np
import pandas as pd
from itertools import product

from src.datasets import SyntheticDataGenerator, MultivariateAnomalyFunction
from src.evaluation.evaluator import Evaluator


# Validates all algorithms regarding polluted data based on a given outlier type.
# The pollution of the training data is tested from 0 to 100% (with default steps=5).
def run_pollution_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5,
                             store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, 'polluted', steps, outlier_type,
                                     store_results=store_results)


# Validates all algorithms regarding missing data based on a given outlier type.
# The percentage of missing values within the training data is tested from 0 to 100% (with default
# steps=5). By default the missing values are represented as zeros since no algorithm can't handle
# nan values.
def run_missing_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5,
                           store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, 'missing', steps, outlier_type,
                                     store_results=store_results)


# high-dimensional experiment on normal outlier types
def run_multi_dim_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5,
                             store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, 'multi_dim', steps, outlier_type,
                                     store_results=store_results)


# Validates all algorithms regarding different heights of extreme outliers
# The extreme values are added to the outlier timestamps everywhere in the dataset distribution.
def run_extremes_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=10,
                            store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, 'extreme', steps, outlier_type,
                                     store_results=store_results)


def run_multivariate_experiment(detectors, seeds, runs, output_dir=None, store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, 'multivariate', store_results=store_results)


def run_multivariate_polluted_experiment(detectors, seeds, runs, outlier_type, output_dir=None, store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, 'mv_polluted',
                                     outlier_type=outlier_type, store_results=store_results)


def run_multi_dim_multivariate_experiment(detectors, seeds, runs, outlier_type, output_dir=None,
                                          steps=2, store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, 'multi_dim_multivariate',
                                     steps, outlier_type=outlier_type, store_results=store_results)


def run_different_window_sizes_evaluator(detectors, seeds, runs):
    results = pd.DataFrame()
    for seed in seeds:
        datasets = [SyntheticDataGenerator.long_term_dependencies_width(seed),
                    SyntheticDataGenerator.long_term_dependencies_height(seed),
                    SyntheticDataGenerator.long_term_dependencies_missing(seed)]
        evaluator = Evaluator(datasets, detectors, seed=seed)
        evaluator.evaluate()
        evaluator.plot_scores()
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)
    evaluator.set_benchmark_results(results)
    evaluator.export_results('run_different_windows')
    evaluator.create_boxplots(runs=runs, data=results, detectorwise=False)
    evaluator.create_boxplots(runs=runs, data=results, detectorwise=True)
    return evaluator


# outlier type means agots types for the univariate experiments, the multivariate types for the multivariate experiments
def get_datasets_for_multiple_runs(anomaly_type, seeds, steps, outlier_type):
    for seed in seeds:
        if anomaly_type == 'extreme':
            yield [SyntheticDataGenerator.get(f'{outlier_type}_extremeness', seed, extreme)
                   for extreme in np.logspace(4, -5, num=steps, base=2)]
        elif anomaly_type == 'missing':
            yield [SyntheticDataGenerator.get(f'{outlier_type}_missing', seed, missing)
                   for missing in np.logspace(-6.5, -0.15, num=steps, base=2)]
        elif anomaly_type == 'polluted':
            yield [SyntheticDataGenerator.get(f'{outlier_type}_polluted', seed, pollution_percentage=pollution)
                   for pollution in [0.01, 0.05, 0.1, 0.2, 0.5]]
        elif anomaly_type == 'mv_polluted':
            yield [MultivariateAnomalyFunction.get_multivariate_dataset(
                outlier_type, random_seed=seed, train_pollution=pollution)
                for pollution in [0.01, 0.05, 0.1, 0.2, 0.5]]
        elif anomaly_type == 'multivariate':
            multivariate_anomaly_functions = ['doubled', 'inversed', 'shrinked', 'delayed', 'xor', 'delayed_missing']
            yield [MultivariateAnomalyFunction.get_multivariate_dataset(dim_func, random_seed=seed)
                   for dim_func in multivariate_anomaly_functions]
        elif anomaly_type == 'multi_dim_multivariate':
            group_sizes = [None, 20]
            num_dims = [25, 75, 125, 250]
            yield [MultivariateAnomalyFunction.get_multivariate_dataset(
                outlier_type, random_seed=seed, features=dim, group_size=gsize,
                name=f'Synthetic Multivariate {dim}-dimensional {outlier_type} '
                     f'Curve Outliers with {gsize or dim} per group'
            )
                for dim, gsize in product(num_dims, group_sizes)]
        elif anomaly_type == 'multi_dim':
            yield [SyntheticDataGenerator.get(f'{outlier_type}', seed, num_dim)
                   for num_dim in np.linspace(100, 1500, steps, dtype=int)]


def run_experiment_evaluation(detectors, seeds, runs, output_dir, anomaly_type, steps=5, outlier_type='extreme_1',
                              store_results=True):
    datasets = list(get_datasets_for_multiple_runs(anomaly_type, seeds, steps, outlier_type))
    results = pd.DataFrame()
    evaluator = None

    for index, seed in enumerate(seeds):
        evaluator = Evaluator(datasets[index], detectors, output_dir, seed=seed)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves(store=store_results)
        evaluator.plot_threshold_comparison(store=store_results)
        evaluator.plot_scores(store=store_results)
        evaluator.set_benchmark_results(result)
        evaluator.export_results(f'experiment-run-{index}-{seed}')
        results = results.append(result, ignore_index=True)

    if not store_results:
        return

    # set average results from multiple pipeline runs for evaluation
    avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results(f'experiment-{anomaly_type}')

    # Plots which need the whole data (not averaged)
    evaluator.create_boxplots(runs=runs, data=results, detectorwise=True, store=store_results)
    evaluator.create_boxplots(runs=runs, data=results, detectorwise=False, store=store_results)
    evaluator.gen_merged_tables(results, f'for_{anomaly_type}', store=store_results)

    # Plots using 'self.benchmark_results' -> using the averaged results
    evaluator.create_bar_charts(runs=runs, detectorwise=True, store=store_results)
    evaluator.create_bar_charts(runs=runs, detectorwise=False, store=store_results)
    evaluator.plot_auroc(title=f'Area under the curve for differing {anomaly_type} anomalies', store=store_results)

    # Plots using 'self.results' (need the score) -> only from the last run
    evaluator.plot_threshold_comparison(store=store_results)
    evaluator.plot_scores(store=store_results)
    evaluator.plot_roc_curves(store=store_results)

    return evaluator


def announce_experiment(title: str, dashes: int = 70):
    print(f'\n###{"-"*dashes}###')
    message = f'Experiment: {title}'
    before = (dashes - len(message)) // 2
    after = dashes - len(message) - before
    print(f'###{"-"*before}{message}{"-"*after}###')
    print(f'###{"-"*dashes}###\n')
