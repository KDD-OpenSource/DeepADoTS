import numpy as np
import pandas as pd

from src.evaluation.evaluator import Evaluator
from src.datasets import SyntheticDataGenerator, MultivariateAnomalyFunction


# Validates all algorithms regarding polluted data based on a given outlier type.
# The pollution of the training data is tested from 0 to 100% (with default steps=5).
def run_pollution_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5,
                             store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "polluted", steps, outlier_type,
                                     store_results=store_results)


# Validates all algorithms regarding missing data based on a given outlier type.
# The percentage of missing values within the training data is tested from 0 to 100% (with default
# steps=5). By default the missing values are represented as zeros since no algorithm can't handle
# nan values.
def run_missing_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5,
                           store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "missing", steps, outlier_type,
                                     store_results=store_results)


# high-dimensional experiment on normal outlier types
def run_multi_dim_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5,
                             store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "multi_dim", steps, outlier_type,
                                     store_results=store_results)


# Validates all algorithms regarding different heights of extreme outliers
# The extreme values are added to the outlier timestamps everywhere in the dataset distribution.
def run_extremes_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=10,
                            store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "extreme", steps, outlier_type,
                                     store_results=store_results)


def run_multivariate_experiment(detectors, seeds, runs, output_dir=None, store_results=True):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "multivariate", store_results=store_results)


def run_multi_dim_multivariate_experiment(detectors, seeds, runs, output_dir=None, steps=2):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "multi_dim_multivariate", steps)


# outlier type means agots types for the univariate experiments, the multivariate types for the multivariate experiments
def get_datasets_for_multiple_runs(anomaly_type, seeds, steps, outlier_type):
    multivariate_anomaly_functions = ['doubled', 'inversed', 'shrinked', 'delayed', 'xor', 'delayed_missing']

    for seed in seeds:
        if anomaly_type == "extreme":
            yield [SyntheticDataGenerator.get(f'{outlier_type}_extremeness', seed, extreme)
                   for extreme in np.linspace(1, 9, steps)]
        elif anomaly_type == "missing":
            yield [SyntheticDataGenerator.get(f'{outlier_type}_missing', seed, missing)
                   for missing in np.linspace(0, 0.9, steps)]
        elif anomaly_type == "polluted":
            yield [SyntheticDataGenerator.get(f'{outlier_type}_polluted', seed, pollution)
                   for pollution in np.linspace(0, 0.5, steps)]
        elif anomaly_type == "multivariate":
            yield [MultivariateAnomalyFunction.get_multivariate_dataset(dim_func, random_seed=seed)
                   for dim_func in multivariate_anomaly_functions]
        elif anomaly_type == "multi_dim_multivariate":
            num_dims = [250, 500, 1000, 1500]
            yield [MultivariateAnomalyFunction.get_multivariate_dataset(outlier_type, random_seed=seed,
                                                                        features=dim, group_size=20,
                   name=f'Synthetic Multivariate {dim}-dimensional {outlier_type} Curve Outliers')
                   for dim in num_dims]
        elif anomaly_type == "multi_dim":
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

    # set average results from multiple pipeline runs for evaluation
    avg_results = results.groupby(["dataset", "algorithm"], as_index=False).mean()
    evaluator.set_benchmark_results(avg_results)
    evaluator.export_results(f'experiment-{anomaly_type}')

    # Plots which need the whole data (not averaged)
    evaluator.create_boxplots(runs=runs, data=results, detectorwise=True, store=store_results)
    evaluator.create_boxplots(runs=runs, data=results, detectorwise=False, store=store_results)
    evaluator.gen_merged_tables(results, f"for_{anomaly_type}", store=store_results)

    # Plots using "self.benchmark_results" -> using the averaged results
    evaluator.create_bar_charts(runs=runs, detectorwise=True, store=store_results)
    evaluator.create_bar_charts(runs=runs, detectorwise=False, store=store_results)
    evaluator.plot_auroc(title=f"Area under the curve for differing {anomaly_type} anomalies", store=store_results)

    # Plots using "self.results" (need the score) -> only from the last run
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
