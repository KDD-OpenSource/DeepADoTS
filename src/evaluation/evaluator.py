import gc
import logging
import os
import pickle
import re
import sys
import traceback
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import progressbar
import time
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import roc_curve, auc
from tabulate import tabulate

from .config import init_logging


class Evaluator:
    def __init__(self, datasets: list, detectors: callable, output_dir: {str} = None, seed: int = None,
                 create_log_file=True):
        """
        :param datasets: list of datasets
        :param detectors: callable that returns list of detectors
        """
        assert np.unique([x.name for x in datasets]).size == len(datasets), 'Some datasets have the same name!'
        self.datasets = datasets
        self._detectors = detectors
        self.output_dir = output_dir or 'reports'
        self.results = dict()
        if create_log_file:
            init_logging(os.path.join(self.output_dir, 'logs'))
        self.logger = logging.getLogger(__name__)
        # Dirty hack: Is set by the main.py to insert results from multiple evaluator runs
        self.benchmark_results = None
        # Last passed seed value in evaluate()
        self.seed = seed

    @property
    def detectors(self):
        detectors = self._detectors(self.seed)
        assert np.unique([x.name for x in detectors]).size == len(detectors), 'Some detectors have the same name!'
        return detectors

    def set_benchmark_results(self, benchmark_result):
        self.benchmark_results = benchmark_result

    def export_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        path = os.path.join(output_dir, f'{name}-{timestamp}.pkl')
        self.logger.info(f'Store evaluator results at {os.path.abspath(path)}')
        save_dict = {
            'datasets': [x.name for x in self.datasets],
            'detectors': [x.name for x in self.detectors],
            'benchmark_results': self.benchmark_results,
            'results': self.results,
            'output_dir': self.output_dir,
            'seed': int(self.seed),
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        return path

    # Import benchmark_results if this evaluator uses the same detectors and datasets
    # self.results are not available because they are overwritten by each run
    def import_results(self, name):
        output_dir = os.path.join(self.output_dir, 'evaluators')
        path = os.path.join(output_dir, f'{name}.pkl')
        self.logger.info(f'Read evaluator results at {os.path.abspath(path)}')
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.logger.debug(f'Importing detectors {"; ".join(save_dict["detectors"])}')
        my_detectors = [x.name for x in self.detectors]
        assert np.array_equal(save_dict['detectors'], my_detectors), 'Detectors should be the same'

        self.logger.debug(f'Importing datasets {"; ".join(save_dict["datasets"])}')
        my_datasets = [x.name for x in self.datasets]
        assert np.array_equal(save_dict['datasets'], my_datasets), 'Datasets should be the same'

        self.benchmark_results = save_dict['benchmark_results']
        self.seed = save_dict['seed']
        self.results = save_dict['results']

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        # warn_for=() avoids log warnings for any result being zero
        precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
        if precision == 0 and recall == 0:
            f01_score = 0
        else:
            f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.1)
        return accuracy, precision, recall, f_score, f01_score

    @staticmethod
    def get_auroc(det, ds, score):
        if np.isnan(score).all():
            score = np.zeros_like(score)
        _, _, _, y_test = ds.data()
        score_nonan = score.copy()
        # Rank NaN below every other value in terms of anomaly score
        score_nonan[np.isnan(score_nonan)] = np.nanmin(score_nonan) - sys.float_info.epsilon
        fpr, tpr, _ = roc_curve(y_test, score_nonan)
        return auc(fpr, tpr)

    def get_optimal_threshold(self, det, y_test, score, steps=100, return_metrics=False):
        maximum = np.nanmax(score)
        minimum = np.nanmin(score)
        threshold = np.linspace(minimum, maximum, steps)
        metrics = list(self.get_metrics_by_thresholds(y_test, score, threshold))
        metrics = np.array(metrics).T
        anomalies, acc, prec, rec, f_score, f01_score = metrics
        if return_metrics:
            return anomalies, acc, prec, rec, f_score, f01_score, threshold
        else:
            return threshold[np.argmax(f_score)]

    def evaluate(self):
        for ds in progressbar.progressbar(self.datasets):
            (X_train, y_train, X_test, y_test) = ds.data()
            for det in progressbar.progressbar(self.detectors):
                self.logger.info(f'Training {det.name} on {ds.name} with seed {self.seed}')
                try:
                    det.fit(X_train.copy())
                    score = det.predict(X_test.copy())
                    self.results[(ds.name, det.name)] = score
                    try:
                        self.plot_details(det, ds, score)
                    except Exception:
                        pass
                except Exception as e:
                    self.logger.error(f'An exception occurred while training {det.name} on {ds}: {e}')
                    self.logger.error(traceback.format_exc())
                    self.results[(ds.name, det.name)] = np.zeros_like(y_test)
            gc.collect()

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                y_pred = self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))
                acc, prec, rec, f1_score, f01_score = self.get_accuracy_precision_recall_fscore(y_test, y_pred)
                score = self.results[(ds.name, det.name)]
                auroc = self.get_auroc(det, ds, score)
                df = df.append({'dataset': ds.name,
                                'algorithm': det.name,
                                'accuracy': acc,
                                'precision': prec,
                                'recall': rec,
                                'F1-score': f1_score,
                                'F0.1-score': f01_score,
                                'auroc': auroc},
                               ignore_index=True)
        return df

    def get_metrics_by_thresholds(self, y_test: list, score: list, thresholds: list):
        for threshold in thresholds:
            anomaly = self.binarize(score, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_test, anomaly)
            yield (anomaly.sum(), *metrics)

    def plot_scores(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            X_train, y_train, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            fig.canvas.set_window_title(ds.name)

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 1)
            sp.set_title('original training data', loc=subtitle_loc)
            for col in X_train.columns:
                plt.plot(X_train[col])
            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 2)
            sp.set_title('original test set', loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])

            sp = fig.add_subplot((2 * len(detectors) + 3), 1, 3)
            sp.set_title('binary labels of test set', loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 4
            for det in detectors:
                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'scores of {det.name}', loc=subtitle_loc)
                score = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(score)), [x for x in score])
                threshold_line = len(score) * [self.get_optimal_threshold(det, y_test, np.array(score))]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2 * len(detectors) + 3), 1, subplot_num)
                sp.set_title(f'binary labels of {det.name}', loc=subtitle_loc)
                plt.plot(np.arange(len(score)),
                         [x for x in self.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))])
                subplot_num += 1
            fig.subplots_adjust(top=0.9, hspace=0.4)
            fig.tight_layout()
            if store:
                self.store(fig, f'scores_{ds.name}')
            figures.append(fig)
        return figures

    def plot_threshold_comparison(self, steps=40, store=True):
        detectors = self.detectors
        plt.close('all')
        plots_shape = len(detectors), len(self.datasets)
        fig, axes = plt.subplots(*plots_shape, figsize=(len(detectors) * 15, len(self.datasets) * 5))
        # Ensure two dimensions for iteration
        axes = np.array(axes).reshape(*plots_shape).T
        plt.suptitle('Compare thresholds', fontsize=10)
        for ds, axes_row in zip(self.datasets, axes):
            _, _, X_test, y_test = ds.data()

            for det, ax in zip(detectors, axes_row):
                score = np.array(self.results[(ds.name, det.name)])

                anomalies, _, prec, rec, f_score, f01_score, thresh = self.get_optimal_threshold(
                    det, y_test, score, return_metrics=True)

                ax.plot(thresh, anomalies / len(y_test),
                        label=fr'anomalies ({len(y_test)} $\rightarrow$ 1)')
                ax.plot(thresh, prec, label='precision')
                ax.plot(thresh, rec, label='recall')
                ax.plot(thresh, f_score, label='f_score', linestyle='dashed')
                ax.plot(thresh, f01_score, label='f01_score', linestyle='dashed')
                ax.set_title(f'{det.name} on {ds.name}')
                ax.set_xlabel('Threshold')
                ax.legend()

        # Avoid overlapping title and axis labels
        plt.xlim([0.0, 1.0])
        fig.subplots_adjust(top=0.9, hspace=0.4, right=1, left=0)
        fig.tight_layout()
        if store:
            self.store(fig, 'metrics_by_thresholds')
        return fig

    def plot_roc_curves(self, store=True):
        detectors = self.detectors
        plt.close('all')
        figures = []
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            fig_scale = 3
            fig = plt.figure(figsize=(fig_scale * len(detectors), fig_scale))
            fig.canvas.set_window_title(ds.name + ' ROC')
            fig.suptitle(f'ROC curve on {ds.name}', fontsize=14, y='1.1')
            subplot_count = 1
            for det in detectors:
                self.logger.info(f'Plotting ROC curve for {det.name} on {ds.name}')
                score = self.results[(ds.name, det.name)]
                if np.isnan(score).all():
                    score = np.zeros_like(score)
                # Rank NaN below every other value in terms of anomaly score
                score[np.isnan(score)] = np.nanmin(score) - sys.float_info.epsilon
                fpr, tpr, _ = roc_curve(y_test, score)
                roc_auc = auc(fpr, tpr)
                plt.subplot(1, len(detectors), subplot_count)
                plt.plot(fpr, tpr, color='darkorange',
                         lw=2, label='area = %0.2f' % roc_auc)
                subplot_count += 1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title('\n'.join(wrap(det.name, 20)))
                plt.legend(loc='lower right')
            plt.tight_layout()
            if store:
                self.store(fig, f'roc_{ds.name}')
            figures.append(fig)
        return figures

    def plot_auroc(self, store=True, title='AUROC'):
        plt.close('all')
        self.benchmark_results[['dataset', 'algorithm', 'auroc']].pivot(
            index='algorithm', columns='dataset', values='auroc').plot(kind='bar')
        plt.legend(loc=3, framealpha=0.5)
        plt.xticks(rotation=20)
        plt.ylabel('AUC', rotation='horizontal', labelpad=20)
        plt.title(title)
        plt.ylim(ymin=0, ymax=1)
        plt.tight_layout()
        if store:
            self.store(plt.gcf(), 'auroc', store_in_figures=True)

    def plot_details(self, det, ds, score, store=True):
        if not det.details:
            return
        plt.close('all')
        cmap = plt.get_cmap('inferno')
        _, _, X_test, y_test = ds.data()

        grid = 0
        for value in det.prediction_details.values():
            grid += 1 if value.ndim == 1 else value.shape[0]
        grid += X_test.shape[1]  # data
        grid += 1 + 1  # score and gt

        fig, axes = plt.subplots(grid, 1, figsize=(15, 1.5 * grid))

        i = 0
        c = cmap(i / grid)
        axes[i].set_title('test data')
        for col in X_test.values.T:
            axes[i].plot(col, color=c)
            i += 1
        c = cmap(i / grid)

        axes[i].set_title('test gt data')
        axes[i].plot(y_test.values, color=c)
        i += 1
        c = cmap(i / grid)

        axes[i].set_title('scores')
        axes[i].plot(score, color=c)
        i += 1
        c = cmap(i / grid)

        for key, values in det.prediction_details.items():
            axes[i].set_title(key)
            if values.ndim == 1:
                axes[i].plot(values, color=c)
                i += 1
            elif values.ndim == 2:
                for v in values:
                    axes[i].plot(v, color=c)
                    i += 1
            else:
                self.logger.warning('plot_details: not sure what to do')
            c = cmap(i / grid)

        fig.tight_layout()
        if store:
            self.store(fig, f'details_{det.name}_{ds.name}')
        return fig

    # create boxplot diagrams for auc values for each algorithm/dataset per algorithm/dataset
    def create_boxplots(self, runs, data, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = data[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].boxplot(by=grouped_by, figsize=(15, 15))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC grouped by {grouped_by} for {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'boxplot_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    # create bar charts for averaged pipeline results per algorithm/dataset
    def create_bar_charts(self, runs, detectorwise=True, store=True):
        target = 'algorithm' if detectorwise else 'dataset'
        grouped_by = 'dataset' if detectorwise else 'algorithm'
        relevant_results = self.benchmark_results[['algorithm', 'dataset', 'auroc']]
        figures = []
        for det_or_ds in (self.detectors if detectorwise else self.datasets):
            relevant_results[relevant_results[target] == det_or_ds.name].plot(x=grouped_by, kind='bar', figsize=(7, 7))
            plt.suptitle('')  # boxplot() adds a suptitle
            plt.title(f'AUC for {target} {det_or_ds.name} over {runs} runs')
            plt.ylim(ymin=0, ymax=1)
            plt.tight_layout()
            figures.append(plt.gcf())
            if store:
                self.store(plt.gcf(), f'barchart_auc_for_{det_or_ds.name}_{runs}_runs', store_in_figures=True)
        return figures

    def store(self, fig, title, extension='pdf', no_counters=False, store_in_figures=False):
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')
        if store_in_figures:
            output_dir = os.path.join(self.output_dir, 'figures')
        else:
            output_dir = os.path.join(self.output_dir, 'figures', f'seed-{self.seed}')
        os.makedirs(output_dir, exist_ok=True)
        counters_str = '' if no_counters else f'-{len(self.detectors)}-{len(self.datasets)}'
        path = os.path.join(output_dir, f'{title}{counters_str}-{timestamp}.{extension}')
        fig.savefig(path)
        self.logger.info(f'Stored plot at {path}')

    def store_text(self, content, title, extension='txt'):
        timestamp = int(time.time())
        output_dir = os.path.join(self.output_dir, 'tables', f'seed-{self.seed}')
        path = os.path.join(output_dir, f'{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        self.logger.info(f'Stored {extension} file at {path}')

    def print_merged_table_per_dataset(self, results):
        for ds in self.datasets:
            table = tabulate(results[results['dataset'] == ds.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Dataset: {ds.name}\n{table}')

    def gen_merged_latex_per_dataset(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for ds in self.datasets:
            content += f'''{ds.name}:\n\n{tabulate(results[results['dataset'] == ds.name],
                                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    def print_merged_table_per_algorithm(self, results):
        for det in self.detectors:
            table = tabulate(results[results['algorithm'] == det.name], headers='keys', tablefmt='psql')
            self.logger.info(f'Detector: {det.name}\n{table}')

    def gen_merged_latex_per_algorithm(self, results, title_suffix=None, store=True):
        title = f'latex_merged{f"_{title_suffix}" if title_suffix else ""}'
        content = ''
        for det in self.detectors:
            content += f'''{det.name}:\n\n{tabulate(results[results['algorithm'] == det.name],
                                   headers='keys', tablefmt='latex')}\n\n'''
        if store:
            self.store_text(content=content, title=title, extension='tex')
        return content

    @staticmethod
    def translate_var_key(key_name):
        if key_name == 'pol':
            return 'Pollution'
        if key_name == 'mis':
            return 'Missing'
        if key_name == 'extremeness':
            return 'Extremeness'
        if key_name == 'f':
            return 'Multivariate'
        # self.logger('Unexpected dataset name (unknown variable in name)')
        return None

    @staticmethod
    def get_key_and_value(dataset_name):
        # Extract var name and value from dataset name
        var_re = re.compile(r'.+\((\w*)=(.*)\)')
        # e.g. 'Syn Extreme Outliers (pol=0.1)'
        match = var_re.search(dataset_name)
        if not match:
            # self.logger.warn('Unexpected dataset name (not variable in name)')
            return '-', dataset_name
        var_key = match.group(1)
        var_value = match.group(2)
        return Evaluator.translate_var_key(var_key), var_value

    @staticmethod
    def get_dataset_types(mi_df):
        types = mi_df.index.get_level_values('Type')
        indexes = np.unique(types, return_index=True)[1]
        return [types[index] for index in sorted(indexes)]

    @staticmethod
    def insert_multi_index_yaxis(ax, mi_df):
        type_title_offset = -1.6  # depends on string length of xaxis ticklabels

        datasets = mi_df.index
        dataset_types = Evaluator.get_dataset_types(mi_df)  # Returns unique entries keeping original order
        logging.getLogger(__name__).debug('Plotting heatmap for groups {" ".join(dataset_types)}')

        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([x[1] for x in datasets])

        y_axis_title_pos = 0  # Store at which position we are for plotting the next title
        for idx, dataset_type in enumerate(dataset_types):
            section_frame = mi_df.iloc[mi_df.index.get_level_values('Type') == dataset_type]
            # Somehow it's sorted by its occurence (which is what we want here)
            dataset_levels = section_frame.index.remove_unused_levels().levels[1]
            title_pos = y_axis_title_pos + 0.5 * (len(dataset_levels) - 1)
            ax.text(type_title_offset, title_pos, dataset_type, ha='center', va='center', rotation=90,
                    fontproperties=FontProperties(weight='bold'))
            if idx < len(dataset_types) - 1:
                sep_pos = y_axis_title_pos + (len(dataset_levels) - 0.6)
                ax.text(-0.5, sep_pos, '_' * int(type_title_offset * -10), ha='right', va='center')
            y_axis_title_pos += len(dataset_levels)

    @staticmethod
    def to_multi_index_frame(evaluators):
        evaluator = evaluators[0]
        for other_evaluator in evaluators[1:]:
            assert evaluator.detectors == other_evaluator.detectors, 'All evaluators should use the same detectors'
        pivot_benchmarks = [ev.benchmark_results.pivot(index='dataset', columns='algorithm',
                                                       values='auroc') for ev in evaluators]

        concat_benchmarks = pd.concat(pivot_benchmarks)
        auroc_matrix = concat_benchmarks.groupby(['dataset']).mean()

        datasets = [[evaluator.get_key_and_value(str(d)) for d in ev.index.values]
                    for ev in pivot_benchmarks]
        datasets = [tuple(d) for d in np.concatenate(datasets)]  # Required for MultiIndex.from_tuples
        datasets = pd.MultiIndex.from_tuples(datasets, names=['Type', 'Level'])
        auroc_matrix.index = datasets
        return auroc_matrix

    def get_multi_index_dataframe(self):
        return Evaluator.to_multi_index_frame([self])

    @staticmethod
    def plot_heatmap(evaluators, store=True):
        mi_df = Evaluator.to_multi_index_frame(evaluators)
        detectors, datasets = mi_df.columns, mi_df.index

        fig, ax = plt.subplots(figsize=(len(detectors) + 2, len(datasets)))
        im = ax.imshow(mi_df, cmap=plt.get_cmap('YlOrRd'), vmin=0, vmax=1)
        plt.colorbar(im)

        # Show MultiIndex for ordinate
        Evaluator.insert_multi_index_yaxis(ax, mi_df)

        # Rotate the tick labels and set their alignment.
        ax.set_xticks(np.arange(len(detectors)))
        ax.set_xticklabels(detectors)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Loop over data dimensions and create text annotations.
        for i in range(len(detectors)):
            for j in range(len(datasets)):
                ax.text(i, j, f'{mi_df.iloc[j, i]:.2f}', ha='center', va='center', color='w',
                        path_effects=[path_effects.withSimplePatchShadow(
                            offset=(1, -1), shadow_rgbFace='b', alpha=0.9)])

        ax.set_title('AUROC over all datasets and detectors')
        # Prevent bug where x axis ticks are completely outside of bounds (matplotlib/issues/5456)
        if len(datasets) > 2:
            fig.tight_layout()
        if store:
            evaluators[0].store(fig, 'heatmap', no_counters=True, store_in_figures=True)
        return fig

    def plot_single_heatmap(self, store=True):
        Evaluator.plot_heatmap([self], store)

    @staticmethod
    def get_printable_runs_results(results):
        print_order = ['dataset', 'algorithm', 'accuracy', 'precision', 'recall', 'F1-score', 'F0.1-score', 'auroc']
        rename_columns = [col for col in print_order if col not in ['dataset', 'algorithm']]

        # calc std and mean for each algorithm per dataset
        std_results = results.groupby(['dataset', 'algorithm']).std(ddof=0).fillna(0)
        # get rid of multi-index
        std_results = std_results.reset_index()
        std_results = std_results[print_order]
        std_results.rename(inplace=True, index=str,
                           columns=dict([(old_col, old_col + '_std') for old_col in rename_columns]))

        avg_results = results.groupby(['dataset', 'algorithm'], as_index=False).mean()
        avg_results = avg_results[print_order]

        avg_results_renamed = avg_results.rename(
            index=str, columns=dict([(old_col, old_col + '_avg') for old_col in rename_columns]))
        return std_results, avg_results, avg_results_renamed

    def gen_merged_tables(self, results, title_suffix=None, store=True):
        title_suffix = f'_{title_suffix}' if title_suffix else ''
        std_results, avg_results, avg_results_renamed = Evaluator.get_printable_runs_results(results)

        ds_title_suffix = f'per_dataset{title_suffix}'
        self.print_merged_table_per_dataset(std_results)
        self.gen_merged_latex_per_dataset(std_results, f'std_{ds_title_suffix}', store=store)

        self.print_merged_table_per_dataset(avg_results_renamed)
        self.gen_merged_latex_per_dataset(avg_results_renamed, f'avg_{ds_title_suffix}', store=store)

        det_title_suffix = f'per_algorithm{title_suffix}'
        self.print_merged_table_per_algorithm(std_results)

        self.gen_merged_latex_per_algorithm(std_results, f'std_{det_title_suffix}', store=store)
        self.print_merged_table_per_algorithm(avg_results_renamed)
        self.gen_merged_latex_per_algorithm(avg_results_renamed, f'avg_{det_title_suffix}', store=store)

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self.threshold(score)
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2 * np.nanstd(score)
