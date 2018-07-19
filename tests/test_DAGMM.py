import unittest

import pandas as pd

from src.algorithms.dagmm import DAGMM
from src.datasets.kdd_cup import KDDCup
from src.evaluation.evaluator import Evaluator


class DAGMMTestCase(unittest.TestCase):
    def test_kdd_cup(self):
        evaluator = Evaluator([KDDCup()], [DAGMM(num_epochs=10, sequence_length=1)])
        df_evaluation = pd.DataFrame(
            columns=['dataset', 'algorithm', 'accuracy', 'precision', 'recall', 'F1-score', 'F0.1-score'])
        for _ in range(5):
            evaluator.evaluate()
            df = evaluator.benchmarks()
            df_evaluation = df_evaluation.append(df)
        print(df_evaluation.to_string())
        assert (df_evaluation == 0).sum().sum() == 0  # No zeroes in the DataFrame
        assert df_evaluation['F1-score'].std() > 0  # Not always the same value
        # Values reported in the paper -1% each
        assert df_evaluation['precision'].mean() >= 0.91
        assert df_evaluation['recall'].mean() >= 0.93
        assert df_evaluation['F1-score'].mean() >= 0.92
