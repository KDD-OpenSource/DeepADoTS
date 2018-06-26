import numpy as np
import pandas as pd

from src.evaluation.evaluator import Evaluator
from src.datasets import SyntheticDataGenerator, MultivariateAnomalyFunction


# Validates all algorithms regarding polluted data based on a given outlier type.
# The pollution of the training data is tested from 0 to 100% (with default steps=5).
def run_pollution_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5):

    results = pd.DataFrame()
    evaluator = None

    for seed in seeds:
        datasets = [
            SyntheticDataGenerator.get(f'{outlier_type}_polluted', seed, pollution) for pollution in
            np.linspace(0, 0.5, steps)
        ]
        evaluator = Evaluator(datasets, detectors)
        evaluator.evaluate(seed)
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

    run_experiment_evaluation(evaluator, runs, results, title='Area under the curve for polluted data')


# Validates all algorithms regarding missing data based on a given outlier type.
# The percentage of missing values within the training data is tested from 0 to 100% (with default
# steps=5). By default the missing values are represented as zeros since no algorithm can't handle
# nan values.
def run_missing_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5):
    results = pd.DataFrame()
    evaluator = None

    for seed in seeds:
        datasets = [
            SyntheticDataGenerator.get(f'{outlier_type}_missing', seed, missing) for missing in
            np.linspace(0, 0.9, steps)
        ]
        evaluator = Evaluator(datasets, detectors)
        evaluator.evaluate(seed)
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

    run_experiment_evaluation(evaluator, runs, results, title='Area under the curve for missing values')


# Validates all algorithms regarding different heights of extreme outliers
# The extreme values are added to the outlier timestamps everywhere in the dataset distribution.
def run_extremes_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=10):
    results = pd.DataFrame()
    evaluator = None

    for seed in seeds:
        datasets = [
            SyntheticDataGenerator.get(f'{outlier_type}_extremeness', seed, extreme) for extreme in
            np.linspace(1, 10, steps)
        ]
        evaluator = Evaluator(datasets, detectors)
        evaluator.evaluate(seed)
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

    run_experiment_evaluation(evaluator, runs, results, title='Area under the curve for differing outlier heights')


def run_multivariate_experiment(detectors, seeds, runs, output_dir=None):
    anomaly_functions = ['doubled', 'inversed', 'shrinked', 'delayed', 'xor']
    results = pd.DataFrame()
    evaluator = None

    for seed in seeds:
        datasets = [
            MultivariateAnomalyFunction.get_multivariate_dataset(dim_func, seed) for dim_func in anomaly_functions
        ]
        evaluator = Evaluator(datasets, detectors)
        evaluator.evaluate(seed)
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

    run_experiment_evaluation(evaluator, runs, results, title='Area under the curve for multivariate outliers')


def run_experiment_evaluation(evaluator=None, runs=None, results=None, outlier_type='extreme_1', title=None):
    evaluator.create_boxplots_per_algorithm(runs=runs, data=results)
    evaluator.create_boxplots_per_dataset(runs=runs, data=results)

    averaged_results = results.groupby(["dataset", "algorithm"], as_index=False).mean()
    evaluator.benchmark_results = averaged_results

    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    evaluator.create_bar_charts_per_dataset(runs=runs)
    evaluator.create_bar_charts_per_algorithm(runs=runs)
    evaluator.generate_latex()
    evaluator.plot_auroc(title=title)
    return evaluator
