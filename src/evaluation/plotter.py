import re
import os
import pickle
import logging
import time

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd

from src.evaluation import Evaluator, LatexGenerator


# For supporting pickles from old versions we need to map them
NAMES_TRANSLATION = {
    'DAGMM_NNAutoEncoder_withWindow': 'DAGMM',
    'DAGMM-NW': 'DAGMM',
    'DAGMM_NNAutoEncoder_withoutWindow': None,
    'DAGMM-NN': None,
    'DAGMM_LSTMAutoEncoder_withWindow': 'LSTM-DAGMM',
    'DAGMM-LW': 'LSTM-DAGMM',
    'LSTMED': 'LSTM-ED',
    'Recurrent EBM': 'REBM',
    'AutoEncoder': 'AE',
}

POLLUTION_REGEX = r'^([^(]+)\(pol=(\d\.\d+), anom=(\d\.\d+)\)'


class Plotter:
    def __init__(self, output_dir, pickle_dirs=None, dataset_names=None, detector_names=[]):
        self.output_dir = output_dir
        self.dataset_names = dataset_names
        self.detector_names = [x for x in detector_names if NAMES_TRANSLATION.get(x, x) is not None]
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
                # ignore hidden files, e.g. .DS_Store on macOS
                if not path.startswith('.'):
                    self.logger.debug("Importing evaluator from '{path}'")
                    with open(os.path.join(dir_, 'evaluators', path), 'rb') as f:
                        save_dict = pickle.load(f)
                    benchmark_results = save_dict['benchmark_results']
                    assert self.dataset_names is None or np.array_equal(self.dataset_names, save_dict['datasets']), \
                        'Runs should be executed on same datasets'
                    self.dataset_names = save_dict['datasets']
                    self.detector_names.extend(save_dict['detectors'])
                    self.detector_names = [x for x in set(self.detector_names)
                                           if NAMES_TRANSLATION.get(x, x) is not None]
                    if benchmark_results is not None:
                        all_results.append(benchmark_results)
                    else:
                        self.logger.warn('benchmark_results was None')
        return all_results

    # --- Final plot functions ----------------------------------------------- #

    # results is an array of benchmark_results (e.g. returned by get_results_for_runs)
    def barplots(self, title):
        plt.close('all')
        if len(self.detector_names) == 1:
            return self.single_barplot(title)
        aurocs = [x[['algorithm', 'dataset', 'auroc']] for x in self.results]
        aurocs_df = pd.concat(aurocs, axis=0, ignore_index=True)

        fig, axes = plt.subplots(
            ncols=len(self.detector_names), figsize=(1.5*len(self.detector_names), 4), sharey=True)
        for ax, det in zip(axes.flat, self.detector_names):
            self._styled_boxplot(ax, aurocs_df, det)

        fig.subplots_adjust(wspace=0)
        fig.suptitle(f'Area under ROC for {title} (runs={len(self.results)})')
        self.store(fig, f'boxplot-experiment-{title}', 'pdf', bbox_inches='tight')
        return fig

    # We only use latex code for generating lineplotsin the paper
    def lineplot(self, title, xlabel=''):
        self.logger.warn('For the paper please use the latex visualization')
        plt.close('all')
        aurocs = self._get_grouped_results()

        fig, ax = plt.subplots(figsize=(4, 4))
        for det in self.detector_names:
            final_det_name = NAMES_TRANSLATION.get(det, det)
            ax.plot([aurocs.loc[ds, det] for ds in self.dataset_names], label=final_det_name)

        ax.set_xticks(list(range(len(self.dataset_names))))
        ax.set_xticklabels([f'{float(Evaluator.get_key_and_value(x)[1]):.2f}' for x in self.dataset_names])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('AUROC')
        ax.set_ylim((0, 1.05))
        ax.yaxis.grid(True)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(wspace=0)
        fig.suptitle(f'Area under ROC for {title} (runs={len(self.results)})')
        self.store(fig, f'lineplot-experiment-{title}', 'pdf', bbox_inches='tight')
        pass

    def latex_lineplot(self, title, **kwargs):
        aurocs = self._get_grouped_results()
        x_ticks = [f'{float(Evaluator.get_key_and_value(x)[1]):.2f}' for x in self.dataset_names]
        lines = [[(pol_value, aurocs.loc[ds, det])
                  for pol_value, ds in zip(x_ticks, self.dataset_names)]
                 for det in self.detector_names]
        latex_output = LatexGenerator.generate_lineplot(title, x_ticks, self.detector_names, lines, **kwargs)
        self.store(latex_output, f'lineplot-{title}', extension='tex')
        return latex_output

    def heatmap(self, title):
        self.logger.warn('Final heatmap function is not implemented')
        pass

    # Can only be used for the pollution experiment where we have two axes for
    # dataset parameters (train pollution, test anomaly percentage)
    def algorithm_heatmaps(self, title):
        plt.close('all')
        aurocs = [x[['algorithm', 'dataset', 'auroc']] for x in self.results]
        aurocs_df = pd.concat(aurocs, axis=0, ignore_index=True)
        det_groups = aurocs_df.groupby('algorithm')

        for det in self.detector_names:
            fig, ax = plt.subplots(figsize=(4, 4))
            det_values = det_groups.get_group(det).drop(columns='algorithm')
            auroc_per_ds = det_values.groupby('dataset').mean().reset_index()
            runs = len(det_values) // len(auroc_per_ds)

            auroc_matrix = Plotter.transform_to_pollution_matrix(auroc_per_ds)

            im = ax.imshow(auroc_matrix, cmap=plt.get_cmap('YlOrRd'), vmin=0, vmax=1)
            plt.colorbar(im)

            ax.set_xticks(np.arange(len(auroc_matrix.columns)))
            ax.set_xticklabels(auroc_matrix.columns)
            ax.set_xlabel('Anomaly percentage')
            # plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            ax.set_yticks(np.arange(len(auroc_matrix.index)))
            ax.set_yticklabels(auroc_matrix.index)
            ax.set_ylabel('Training pollution')

            # Loop over data dimensions and create text annotations.
            for i in range(len(auroc_matrix.columns)):
                for j in range(len(auroc_matrix.index)):
                    ax.text(i, j, f'{auroc_matrix.iloc[j, i]:.2f}', ha='center', va='center', color='w',
                            path_effects=[path_effects.withSimplePatchShadow(
                                offset=(1, -1), shadow_rgbFace='b', alpha=0.9)])

            final_det_name = NAMES_TRANSLATION.get(det, det)
            fig.suptitle(f'{title.title()} (runs={runs})')
            ax.set_title(f'AUROC for {final_det_name}')
            self.store(fig, f'heatmap-{final_det_name}-{title}', 'pdf', bbox_inches='tight')

    # --- Helper functions --------------------------------------------------- #

    def _get_grouped_results(self):
        results = pd.DataFrame(index=self.dataset_names, columns=self.detector_names)
        aurocs = [x[['algorithm', 'dataset', 'auroc']] for x in self.results]
        aurocs_df = pd.concat(aurocs, axis=0, ignore_index=True)
        for det in self.detector_names:
            values = aurocs_df[aurocs_df['algorithm'] == det].drop(columns='algorithm')
            ds_groups = values.groupby('dataset')
            for ds in self.dataset_names:
                results.loc[ds, det] = ds_groups.get_group(ds)['auroc'].mean()
        return results

    @staticmethod
    def transform_to_pollution_matrix(auroc_per_ds_avg):
        # Aurocs should be already averaged for each dataset
        # Example ds name: Syn Extreme Outliers (pol=0.3, anom=0.5)
        auroc_matrix = pd.DataFrame(
            auroc_per_ds_avg.dataset
            # Remove dataset name
            .str.replace(POLLUTION_REGEX, '\\2-\\3')
            # Extract values
            .str.split('-')
            .tolist(), columns=['pol', 'anom']) \
            .assign(auroc=auroc_per_ds_avg.auroc.values) \
            .pivot(index='pol', columns='anom', values='auroc')
        return auroc_matrix.reindex(index=auroc_matrix.index[::-1])

    # Selects the greatest anomaly percentage and thereby filters entries
    def fix_anomaly_percentage(self, anom_perc_idx=-2):
        anom_percs = sorted(self.results[0].dataset.str.replace(POLLUTION_REGEX, '\\3').astype(float).unique())
        sel_anom = anom_percs[anom_perc_idx]  # max(anom_percs)
        for i in range(len(self.results)):
            # Filter out results for other anomoly_percentage values
            self.results[i] = self.results[i][self.results[i]['dataset'].str.contains(f'anom={sel_anom}')]
            # Rename dataset names so they don't contain the anomoly_percentage anymore
            self.results[i].loc[:, 'dataset'] = self.results[i]['dataset'].str.replace(POLLUTION_REGEX, '\\1(pol=\\2)')
            self.results[i].loc[:, 'dataset'] = [
                self._use_absolute_pollution(x, sel_anom) for x in self.results[i]['dataset']]
        # Adapt dataset names accordingly
        self.dataset_names = self.results[0].dataset.unique()
        self.dataset_names = sorted(self.dataset_names, key=lambda x: float(Evaluator.get_key_and_value(x)[1]))
        return sel_anom

    def _use_absolute_pollution(self, dataset_name, anomaly_percentage):
        name_regex = re.compile(r'^([^(]+)\(pol=(\d\.\d+)\)')
        match = name_regex.match(dataset_name)
        assert match is not None, 'Dataset needs to be polluted'
        rel_pollution = float(match.group(2))
        # abs_pollution = rel_pollution * anomaly_percentage
        abs_pollution = rel_pollution
        return f'{match.group(1)}(pol={abs_pollution})'

    def single_barplot(self, title):
        aurocs = [x[['algorithm', 'dataset', 'auroc']] for x in self.results]
        aurocs_df = pd.concat(aurocs, axis=0, ignore_index=True)
        det = self.detector_names[0]

        fig, ax = plt.subplots(figsize=(4, 4))
        self._styled_boxplot(ax, aurocs_df, det)
        ax.set_xlabel(None)

        final_det_name = NAMES_TRANSLATION.get(det, det)
        fig.subplots_adjust(wspace=0)
        fig.suptitle(f'Area under ROC for {final_det_name} (runs={len(self.results)})')
        self.store(fig, f'boxplot-{final_det_name}-{title}', 'pdf', bbox_inches='tight')
        return fig

    def _styled_boxplot(self, ax, aurocs_df, det):
        values = aurocs_df[aurocs_df['algorithm'] == det].drop(columns='algorithm')
        ds_groups = values.groupby('dataset')
        ax.boxplot([ds_groups.get_group(x)['auroc'].values for x in self.dataset_names],
                   positions=np.linspace(0, 1, len(self.dataset_names)), widths=0.15,
                   medianprops={'linewidth': 1},
                   whiskerprops={'color': 'darkblue', 'linestyle': '--'},
                   flierprops={'markersize': 3},
                   boxprops={'color': 'darkblue'})
        ax.set_xticklabels([f'{float(Evaluator.get_key_and_value(x)[1]):.2f}' for x in self.dataset_names],
                           rotation=90)

        ax.set_xlabel(NAMES_TRANSLATION.get(det, det))
        ax.set_ylim((0, 1.05))
        ax.yaxis.grid(True)
        ax.set_xlim((-0.15, 1.15))
        ax.margins(0.05)

    def store(self, fig_or_content, title, extension='pdf', **kwargs):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        output_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{title}-{timestamp}.{extension}')
        if extension in ('pdf', 'png'):
            fig_or_content.savefig(path, **kwargs)
        else:
            with open(path, 'w') as f:
                f.write(fig_or_content)
        self.logger.info(f'Stored plot at {path}')
