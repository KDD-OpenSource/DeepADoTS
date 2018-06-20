import os

import numpy as np
import pandas as pd

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTMAD, LSTM_Enc_Dec
from src.datasets import AirQuality, KDDCup, SyntheticDataGenerator
from src.evaluation.evaluator import Evaluator
from experiments import run_pollution_experiment, run_missing_experiment, run_extremes_experiment, \
                        run_multivariate_experiment


def main():
    run_pipeline()
    run_experiments()


def run_pipeline():
    if os.environ.get("CIRCLECI", False):
        datasets = [SyntheticDataGenerator.extreme_1()]
        detectors = [RecurrentEBM(num_epochs=2), LSTMAD(num_epochs=5), Donut(num_epochs=5), DAGMM(),
                     LSTM_Enc_Dec(num_epochs=2)]
    else:
        datasets = [
            SyntheticDataGenerator.extreme_1(),
            SyntheticDataGenerator.variance_1(),
            SyntheticDataGenerator.shift_1(),
            SyntheticDataGenerator.trend_1(),
            SyntheticDataGenerator.combined_1(),
            SyntheticDataGenerator.combined_4(),
        ]
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(), Donut(), DAGMM(), LSTM_Enc_Dec(num_epochs=15)]
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    '''evaluator.print_tables()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()
    evaluator.plot_roc_curves()'''
    evaluator.generate_latex()


def evaluate_on_real_world_data_sets():
    dagmm = DAGMM()
    kdd_cup = KDDCup()
    X_train, y_train, X_test, y_test = kdd_cup.data()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print(Evaluator.get_accuracy_precision_recall_fscore(y_test, pred))

    donut = Donut()
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
    if os.environ.get("CIRCLECI", False):
        run_extremes_experiment(outlier_type, output_dir=os.path.join(output_dir, 'extremes'),
                                steps=1)
    else:
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


if __name__ == '__main__':
    main()
