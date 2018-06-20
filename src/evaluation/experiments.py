import os

import numpy as np

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMAD, LSTM_Enc_Dec
from src.evaluation.evaluator import Evaluator
from src.datasets import SyntheticDataGenerator, MultivariateAnomalyFunction


# Validates all algorithms regarding polluted data based on a given outlier type.
# The pollution of the training data is tested from 0 to 100% (with default steps=5).
def run_pollution_experiment(outlier_type='extreme_1', output_dir=None, steps=5):
    datasets = [
        SyntheticDataGenerator.get(f'{outlier_type}_polluted', pollution) for pollution in np.linspace(0, 1, steps)
    ]
    detectors = [LSTM_Enc_Dec(num_epochs=15), DAGMM(), Donut(), RecurrentEBM(), LSTMAD()]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
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
def run_missing_experiment(outlier_type='extreme_1', output_dir=None, steps=5):
    datasets = [
        SyntheticDataGenerator.get(f'{outlier_type}_missing', missing) for missing in np.linspace(0, 1, steps)
    ]
    detectors = [LSTM_Enc_Dec(num_epochs=200), DAGMM(), Donut(), RecurrentEBM(), LSTMAD()]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
    evaluator.plot_auroc(title='Area under the curve for missing values')
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    return evaluator


# Validates all algorithms regarding different heights of extreme outliers
# The extreme values are added to the outlier timestamps everywhere in the dataset distribution.
def run_extremes_experiment(outlier_type='extreme_1', output_dir=None, steps=10):
    datasets = [
        SyntheticDataGenerator.get(f'{outlier_type}_extremeness', extreme) for extreme in np.linspace(1, 10, steps)
    ]
    detectors = [LSTM_Enc_Dec(num_epochs=200), DAGMM(), Donut(), RecurrentEBM(), LSTMAD()]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
    evaluator.plot_auroc(title='Area under the curve for differing outlier heights')
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    return evaluator


def run_multivariate_experiment(output_dir=None):
    anomaly_functions = ['doubled', 'inversed', 'shrinked', 'delayed', 'xor']
    datasets = [
        MultivariateAnomalyFunction.get_multivariate_dataset(dim_func) for dim_func in anomaly_functions
    ]
    detectors = [RecurrentEBM(num_epochs=5), LSTMAD(num_epochs=5)]
    evaluator = Evaluator(datasets, detectors, output_dir)
    evaluator.evaluate()
    evaluator.plot_auroc(title='Area under the curve for multivariate outliers')
    evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()
    return evaluator


def run_experiments(outlier_type='extreme_1', output_dir=None, steps=5):
    output_dir = output_dir or os.path.join('reports/experiments', outlier_type)

    announce_experiment('Missing Values')
    run_pollution_experiment(outlier_type, output_dir=os.path.join(output_dir, 'pollution'),
                             steps=steps)

    announce_experiment('Pollution')
    run_missing_experiment(outlier_type, output_dir=os.path.join(output_dir, 'missing'),
                           steps=steps)

    announce_experiment('Outlier height')
    run_extremes_experiment(outlier_type, output_dir=os.path.join(output_dir, 'extremes'),
                            steps=steps)

    announce_experiment('Multivariate Datasets')
    run_multivariate_experiment(output_dir=os.path.join(output_dir, 'multivariate'))


def announce_experiment(title: str, dashes: int = 70):
    print(f'\n###{"-"*dashes}###')
    message = f'Experiment: {title}'
    before = (dashes - len(message)) // 2
    after = dashes - len(message) - before
    print(f'###{"-"*before}{message}{"-"*after}###')
    print(f'###{"-"*dashes}###\n')
