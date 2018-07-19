import os
import pickle
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluator import Evaluator


class Plotter:
    def __init__(self, output_dir, pickle_dirs=None, dataset_names=None, detector_names=None):
        self.output_dir = output_dir
        self.dataset_names = dataset_names
        self.detector_names = detector_names
        self.results = None
        self.logger = logging.getLogger(__name__)
        if pickle_dirs is not None:
            self.results = self.import_results_for_runs(pickle_dirs)

    # pickle_dirs is an array of directories where to look for a
    # subdirectory 'evaluators' with stored pickles. The datasets and detectors
    # used for these pickles must match the ones passed to this Plotter instance
    def import_results_for_runs(self, pickle_dirs=['data']):
        if not isinstance(pickle_dirs, list):
            pickle_dirs = [pickle_dirs]
        all_results = []
        for dir_ in pickle_dirs:
            for path in os.listdir(os.path.join(dir_, 'evaluators')):
                with open(os.path.join(dir_, 'evaluators', path), 'rb') as f:
                    save_dict = pickle.load(f)
                benchmark_results = save_dict['benchmark_results']
                assert self.dataset_names is None or np.array_equal(self.dataset_names, save_dict['datasets']), \
                    'Runs should be executed on same datasets'
                self.dataset_names = save_dict['datasets']
                self.detector_names = save_dict['detectors']
                if benchmark_results is not None:
                    all_results.append(benchmark_results)
                else:
                    self.logger.warn('benchmark_results was None')
        return all_results

    # results is an array of benchmark_results (e.g. returned by get_results_for_runs)
    def plot_experiment(self, title):
        aurocs = [x[['algorithm', 'dataset', 'auroc']] for x in self.results]
        aurocs_df = pd.concat(aurocs, axis=0, ignore_index=True)

        fig, axes = plt.subplots(
            ncols=len(self.detector_names), figsize=(3*len(self.detector_names), 4), sharey=True)

        for ax, det in zip(axes.flat, self.detector_names):
            values = aurocs_df[aurocs_df['algorithm'] == det].drop(columns='algorithm')
            ds_groups = values.groupby('dataset')
            ax.boxplot([ds_groups.get_group(x)['auroc'].values for x in self.dataset_names],
                       positions=np.linspace(0, 1, 5))
            ax.set_xticklabels([f'{float(Evaluator.get_key_and_value(x)[1]):.2f}' for x in self.dataset_names])
            
            ax.set_xlabel(det, rotation=15)
            ax.set_ylim((0, 1.05))
            ax.yaxis.grid(True)
            ax.set_xlim((-0.15, 1.15))
            ax.margins(0.05)

        fig.subplots_adjust(wspace=0)
        fig.suptitle(f'Area under ROC for {title} (runs={len(self.results)})')
        self.store(fig, f'boxplot-experiment-{title}.pdf', bbox_inches='tight')
        return fig

    def store(self, fig, title, extension="pdf", **kwargs):
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        output_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{title}-{timestamp}.{extension}")
        fig.savefig(path, **kwargs)
        self.logger.info(f"Stored plot at {path}")
