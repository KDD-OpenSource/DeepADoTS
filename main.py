import os

from tabulate import tabulate

from src.algorithms import DAGMM, Donut, RecurrentEBM, LSTM_Enc_Dec, LSTMAD
from src.datasets.synthetic_data_generator import SyntheticDataGenerator
from src.evaluation.evaluator import Evaluator


def main():
    if os.environ.get("CIRCLECI", False):
        datasets = [SyntheticDataGenerator.extreme_1()]
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(num_epochs=10), Donut(max_epoch=5), DAGMM(),
                     LSTM_Enc_Dec(epochs=10)]
    else:
        datasets = [
            SyntheticDataGenerator.extreme_1(),
            SyntheticDataGenerator.variance_1(),
            SyntheticDataGenerator.shift_1(),
            SyntheticDataGenerator.trend_1(),
            SyntheticDataGenerator.combined_1(),
            SyntheticDataGenerator.combined_4(),
            SyntheticDataGenerator.variance_1_missing(0.1),
            SyntheticDataGenerator.variance_1_missing(0.3),
            SyntheticDataGenerator.variance_1_missing(0.5),
            SyntheticDataGenerator.variance_1_missing(0.8),
            SyntheticDataGenerator.extreme_1_polluted(0.1),
            SyntheticDataGenerator.extreme_1_polluted(0.3),
            SyntheticDataGenerator.extreme_1_polluted(0.5),
            SyntheticDataGenerator.extreme_1_polluted(1)
        ]
        detectors = [RecurrentEBM(num_epochs=15), LSTMAD(), Donut(), DAGMM(), LSTM_Enc_Dec(epochs=200)]
    evaluator = Evaluator(datasets, detectors)
    evaluator.evaluate()
    df = evaluator.benchmarks()
    for ds in df['dataset'].unique():
        print("Dataset: " + ds)
        print_order = ["algorithm", "accuracy", "precision", "recall", "F1-score", "F0.1-score"]
        print(tabulate(df[df['dataset'] == ds][print_order], headers='keys', tablefmt='psql'))
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
