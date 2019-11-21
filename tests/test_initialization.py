"""Test each detector on each synthetic dataset"""

import os
import unittest

import numpy as np

from experiments import run_extremes_experiment, announce_experiment
from src.algorithms import AutoEncoder, DAGMM, RecurrentEBM, LSTMAD, LSTMED


class InitializationTestCase(unittest.TestCase):

    @staticmethod
    def test_algorithm_initializations():
        def detectors(seed):
            dets = [AutoEncoder(num_epochs=1, seed=seed), DAGMM(num_epochs=1, seed=seed),
                    DAGMM(num_epochs=1, autoencoder_type=DAGMM.AutoEncoder.LSTM, seed=seed),
                    LSTMAD(num_epochs=1, seed=seed), LSTMED(num_epochs=1, seed=seed),
                    RecurrentEBM(num_epochs=1, seed=seed)]
            return sorted(dets, key=lambda x: x.framework)

        RUNS = 1
        seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
        output_dir = 'reports/experiments'
        evaluators = []
        outlier_height_steps = 1

        for outlier_type in ['extreme_1', 'shift_1', 'variance_1', 'trend_1']:
            announce_experiment('Outlier Height')
            ev_extr = run_extremes_experiment(
                detectors, seeds, RUNS, outlier_type, steps=outlier_height_steps,
                output_dir=os.path.join(output_dir, outlier_type, 'intensity'))
            evaluators.append(ev_extr)

        ev_extr.plot_single_heatmap()
