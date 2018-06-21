import numpy as np

from src.evaluation.evaluator import Evaluator
from src.datasets import SyntheticDataGenerator, MultivariateAnomalyFunction


# Validates all algorithms regarding polluted data based on a given outlier type.
# The pollution of the training data is tested from 0 to 100% (with default steps=5).
def run_pollution_experiment(detectors, outlier_type='extreme_1', output_dir=None, steps=5):
    datasets = [
        SyntheticDataGenerator.get(f'{outlier_type}_polluted', pollution) for pollution in np.linspace(0, 0.5, steps)
    ]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
    evaluator.benchmark_results = evaluator.benchmarks()
    evaluator.plot_auroc(title='Area under the curve for polluted data')
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    return evaluator


# Validates all algorithms regarding missing data based on a given outlier type.
# The percentage of missing values within the training data is tested from 0 to 100% (with default
# steps=5). By default the missing values are represented as zeros since no algorithm can't handle
# nan values.
def run_missing_experiment(detectors, outlier_type='extreme_1', output_dir=None, steps=5):
    datasets = [
        SyntheticDataGenerator.get(f'{outlier_type}_missing', missing) for missing in np.linspace(0, 0.9, steps)
    ]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
    evaluator.benchmark_results = evaluator.benchmarks()
    evaluator.plot_auroc(title='Area under the curve for missing values')
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    return evaluator


# Validates all algorithms regarding different heights of extreme outliers
# The extreme values are added to the outlier timestamps everywhere in the dataset distribution.
def run_extremes_experiment(detectors, outlier_type='extreme_1', output_dir=None, steps=10):
    datasets = [
        SyntheticDataGenerator.get(f'{outlier_type}_extremeness', extreme) for extreme in np.linspace(1, 10, steps)
    ]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
    evaluator.benchmark_results = evaluator.benchmarks()
    evaluator.plot_auroc(title='Area under the curve for differing outlier heights')
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    return evaluator


def run_multivariate_experiment(detectors, output_dir=None):
    anomaly_functions = ['doubled', 'inversed', 'shrinked', 'delayed', 'xor']
    datasets = [
        MultivariateAnomalyFunction.get_multivariate_dataset(dim_func) for dim_func in anomaly_functions
    ]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
    evaluator.benchmark_results = evaluator.benchmarks()
    evaluator.plot_auroc(title='Area under the curve for multivariate outliers')
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    return evaluator


def run_multid_multivariate_experiment(detectors, output_dir=None, steps=2):
    num_dims = np.linspace(6, 10, steps, dtype=int)
    datasets = [
        MultivariateAnomalyFunction.get_multivariate_dataset('doubled', features=dim) for dim in num_dims
    ]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
    evaluator.benchmark_results = evaluator.benchmarks()
    evaluator.plot_auroc(title='Area under the curve for multivariate outliers')
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    return evaluator
