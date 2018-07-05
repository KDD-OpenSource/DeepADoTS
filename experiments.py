import numpy as np
import pandas as pd

from src.evaluation.evaluator import Evaluator
from src.datasets import SyntheticDataGenerator, MultivariateAnomalyFunction


# Validates all algorithms regarding polluted data based on a given outlier type.
# The pollution of the training data is tested from 0 to 100% (with default steps=5).
def run_pollution_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "polluted", steps, outlier_type)


# Validates all algorithms regarding missing data based on a given outlier type.
# The percentage of missing values within the training data is tested from 0 to 100% (with default
# steps=5). By default the missing values are represented as zeros since no algorithm can't handle
# nan values.
def run_missing_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=5):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "missing", steps, outlier_type)


# Validates all algorithms regarding different heights of extreme outliers
# The extreme values are added to the outlier timestamps everywhere in the dataset distribution.
def run_extremes_experiment(detectors, seeds, runs, outlier_type='extreme_1', output_dir=None, steps=10):
    return run_experiment_evaluation(detectors, seeds, runs, output_dir, "extreme", steps, outlier_type)


def run_multivariate_experiment(detectors, seeds, runs, output_dir=None):
    return run_experiment_evaluation(detectors=detectors, seeds=seeds, runs=runs, output_dir=output_dir,
                                     anomaly_type="multivariate")


def run_experiment_evaluation(detectors, seeds, runs, output_dir, anomaly_type, steps=5, outlier_type='extreme_1'):
    multivariate_anomaly_functions = ['doubled', 'inversed', 'shrinked', 'delayed', 'xor']
    print_order = ["dataset", "algorithm", "accuracy", "precision", "recall", "F1-score", "F0.1-score", "auroc"]
    rename_columns = [col for col in print_order if col not in ['dataset', 'algorithm']]

    data_dict = {
        "extreme": [],
        "missing": [],
        "polluted": [],
        "multivariate": [],
    }

    for seed in seeds:
        if anomaly_type == "extreme":
            data_dict["extreme"].append([SyntheticDataGenerator.get(f'{outlier_type}_extremeness', seed, extreme)
                                         for extreme in np.linspace(1, 9, steps)])
        elif anomaly_type == "missing":
            data_dict["missing"].append([SyntheticDataGenerator.get(f'{outlier_type}_missing', seed, missing)
                                         for missing in np.linspace(0, 0.9, steps)])
        elif anomaly_type == "polluted":
            data_dict["polluted"].append([SyntheticDataGenerator.get(f'{outlier_type}_polluted', seed, pollution)
                                          for pollution in np.linspace(0, 0.5, steps)])
        elif anomaly_type == "multivariate":
            data_dict["multivariate"].append([MultivariateAnomalyFunction.get_multivariate_dataset(dim_func,
                                             random_seed=seed) for dim_func in multivariate_anomaly_functions])

    results = pd.DataFrame()
    evaluator = None

    for index, seed in enumerate(seeds):
        evaluator = Evaluator(data_dict[anomaly_type][index], detectors, output_dir, seed)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        results = results.append(result, ignore_index=True)

    evaluator.create_boxplots_per_algorithm(runs=runs, data=results)
    evaluator.create_boxplots_per_dataset(runs=runs, data=results)

    # calc std and mean for each algorithm per dataset
    std_results = results.groupby(["dataset", "algorithm"]).std(ddof=0)
    # get rid of multi-index
    std_results = std_results.reset_index()
    std_results = std_results[print_order]
    std_results.rename(inplace=True, index=str,
                       columns=dict([(old_col, old_col + '_std') for old_col in rename_columns]))

    avg_results = results.groupby(["dataset", "algorithm"], as_index=False).mean()
    avg_results = avg_results[print_order]

    avg_results_renamed = avg_results.rename(index=str,
                                             columns=dict([(old_col, old_col + '_avg') for old_col in rename_columns]))

    evaluator.print_merged_table_per_dataset(std_results)
    evaluator.gen_latex_for_merged_table_per_dataset(std_results,
                                                     title="latex_merged_std_results__for_{anomaly_type}_anomalies")

    evaluator.print_merged_table_per_dataset(avg_results_renamed)
    evaluator.gen_latex_for_merged_table_per_dataset(avg_results_renamed,
                                                     title="latex_merged_avg_results_for_{anomaly_type}_anomalies")

    evaluator.print_merged_table_per_algorithm(std_results)
    evaluator.gen_latex_for_merged_table_per_algorithm(
        std_results, title=f"latex_merged_std_results_per_algorithm_for_{anomaly_type}_anomalies")

    evaluator.print_merged_table_per_algorithm(avg_results_renamed)
    evaluator.gen_latex_for_merged_table_per_algorithm(
        avg_results_renamed, title="latex_merged_avg_results_per_algorithm_for_{anomaly_type}_anomalies")

    # set average results from multiple pipeline runs for evaluation
    evaluator.benchmark_results = avg_results

    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    evaluator.create_bar_charts_per_dataset(runs=runs)
    evaluator.create_bar_charts_per_algorithm(runs=runs)
    evaluator.plot_auroc(title=f"Area under the curve for differing {anomaly_type} anomalies")
    return evaluator
