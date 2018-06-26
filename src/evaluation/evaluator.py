import logging
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import time
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import roc_curve, auc
from tabulate import tabulate
from textwrap import wrap

from .config import init_logging


class Evaluator:
    def __init__(self, datasets: list, detectors: list, output_dir: {str} = None):
        self.datasets = datasets
        self.detectors = detectors
        self.output_dir = output_dir or 'reports/figures/'
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = dict()
        init_logging(output_dir or 'reports/logs/')
        self.logger = logging.getLogger(__name__)
        self.benchmark_results = None

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        # warn_for=() avoids log warnings for any result being zero
        precision, recall, f_score, _ = prf(y_true, y_pred, average="binary", warn_for=())
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

    def get_optimal_threshold(self, det, y_test, score, steps=40, return_metrics=False):
        maximum = np.nanmax(score)
        minimum = np.nanmin(score)
        threshold = np.linspace(minimum, maximum, steps)
        metrics = list(self.get_metrics_by_thresholds(det, y_test, score, threshold))
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
                self.logger.info(f"Training {det.name} on {ds}")
                try:
                    det.fit(X_train, y_train)
                    score = det.predict(X_test)
                    self.results[(ds.name, det.name)] = score
                except Exception as e:
                    self.logger.error(f"An exception occurred while training {det.name} on {ds}: {e}")
                    self.logger.error(traceback.format_exc())
                    self.results[(ds.name, det.name)] = np.zeros_like(y_test)

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                y_pred = det.binarize(score, self.get_optimal_threshold(det, y_test, np.array(score)))
                acc, prec, rec, f1_score, f01_score = self.get_accuracy_precision_recall_fscore(y_test, y_pred)
                score = self.results[(ds.name, det.name)]
                auroc = self.get_auroc(det, ds, score)
                df = df.append({"dataset": ds.name,
                                "algorithm": det.name,
                                "accuracy": acc,
                                "precision": prec,
                                "recall": rec,
                                "F1-score": f1_score,
                                "F0.1-score": f01_score,
                                "auroc": auroc},
                               ignore_index=True)
        return df

    @staticmethod
    def get_metrics_by_thresholds(det, y_test: list, score: list, thresholds: list):
        for threshold in thresholds:
            anomaly = det.binarize(score, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_test, anomaly)
            yield (anomaly.sum(), *metrics)

    def plot_scores(self, store=True):
        plt.close('all')
        figures = []
        for ds in self.datasets:
            X_train, y_train, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            fig.canvas.set_window_title(ds.name)

            sp = fig.add_subplot((2 * len(self.detectors) + 3), 1, 1)
            sp.set_title("original training data", loc=subtitle_loc)
            for col in X_train.columns:
                plt.plot(X_train[col])
            sp = fig.add_subplot((2 * len(self.detectors) + 3), 1, 2)
            sp.set_title("original test set", loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])

            sp = fig.add_subplot((2 * len(self.detectors) + 3), 1, 3)
            sp.set_title("binary labels of test set", loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 4
            for det in self.detectors:
                sp = fig.add_subplot((2 * len(self.detectors) + 3), 1, subplot_num)
                sp.set_title(f"scores of {det.name}", loc=subtitle_loc)
                score = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(score)), [x for x in score])
                threshold_line = len(score) * [det.threshold(score)]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2 * len(self.detectors) + 3), 1, subplot_num)
                sp.set_title(f"binary labels of {det.name}", loc=subtitle_loc)
                plt.plot(np.arange(len(score)), [x for x in det.binarize(score)])
                subplot_num += 1
            fig.subplots_adjust(top=0.9, hspace=0.4)
            fig.tight_layout()
            if store:
                self.store(fig, f"scores_{ds.name}")
            figures.append(fig)
        return figures

    def plot_threshold_comparison(self, steps=40, store=True):
        plt.close('all')
        plots_shape = len(self.detectors), len(self.datasets)
        fig, axes = plt.subplots(*plots_shape, figsize=(len(self.detectors) * 5, len(self.datasets) * 5))
        # Ensure two dimensions for iteration
        axes = np.array(axes).reshape(*plots_shape).T
        plt.suptitle("Compare thresholds", fontsize=16)
        for ds, axes_row in zip(self.datasets, axes):
            _, _, X_test, y_test = ds.data()

            for det, ax in zip(self.detectors, axes_row):
                score = np.array(self.results[(ds.name, det.name)])

                anomalies, _, prec, rec, f_score, f01_score, thresh = self.get_optimal_threshold(
                    det, y_test, score, return_metrics=True)

                ax.plot(thresh, anomalies / len(y_test),
                        label=fr"anomalies ({len(y_test)} $\rightarrow$ 1)")
                ax.plot(thresh, prec, label="precision")
                ax.plot(thresh, rec, label="recall")
                ax.plot(thresh, f_score, label="f_score", linestyle='dashed')
                ax.plot(thresh, f01_score, label="f01_score", linestyle='dashed')
                ax.set_title(f"{det.name} on {ds.name}")
                ax.set_xlabel("Threshold")
                ax.legend()

        fig.tight_layout()
        # Avoid overlapping title and axis labels
        fig.subplots_adjust(top=0.9, hspace=0.4)
        if store:
            self.store(fig, "metrics_by_thresholds")
        return fig

    def plot_roc_curves(self, store=True):
        plt.close('all')
        figures = []
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            fig_scale = 3
            fig = plt.figure(figsize=(fig_scale * len(self.detectors), fig_scale))
            fig.canvas.set_window_title(ds.name + " ROC")
            fig.suptitle(f"ROC curve on {ds.name}", fontsize=14, y="1.1")
            subplot_count = 1
            for det in self.detectors:
                self.logger.info(f"Plotting ROC curve for {det.name} on {ds.name}")
                score = self.results[(ds.name, det.name)]
                if np.isnan(score).all():
                    score = np.zeros_like(score)
                # Rank NaN below every other value in terms of anomaly score
                score[np.isnan(score)] = np.nanmin(score) - sys.float_info.epsilon
                fpr, tpr, _ = roc_curve(y_test, score)
                roc_auc = auc(fpr, tpr)
                plt.subplot(1, len(self.detectors), subplot_count)
                plt.plot(fpr, tpr, color="darkorange",
                         lw=2, label="area = %0.2f" % roc_auc)
                subplot_count += 1
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.gca().set_aspect("equal", adjustable="box")
                plt.title('\n'.join(wrap(det.name, 20)))
                plt.legend(loc="lower right")
            plt.tight_layout()
            if store:
                self.store(fig, f"roc_{ds.name}")
            figures.append(fig)
        return figures

    def plot_auroc(self, store=True, title='AUROC'):
        plt.close('all')
        detector_names = [det.name for det in self.detectors]
        fig = plt.figure(figsize=(7, 7))
        for index, ds in enumerate(self.datasets):
            aurocs = self.benchmark_results[self.benchmark_results['dataset'] == ds.name]['auroc']
            width = 0.8 / len(self.datasets)
            plt.bar(np.arange(len(aurocs.values)) + index*width, aurocs.values, width=width, label=ds.name)
        plt.xticks(range(len(self.detectors)), detector_names, rotation=90)
        plt.legend(loc=3, framealpha=0.5)
        plt.xlabel('Dataset')
        plt.ylabel('Area under Receiver Operating Characteristic')
        plt.title(title)
        fig.tight_layout()
        if store:
            self.store(fig, f"auroc")
        return fig

    def print_tables(self):
        for ds in self.datasets:
            print_order = ["algorithm", "accuracy", "precision", "recall", "F1-score", "F0.1-score"]
            table = tabulate(self.benchmark_results[self.benchmark_results['dataset'] == ds.name][print_order],
                             headers='keys', tablefmt='psql')
            self.logger.info(f"Dataset: {ds.name}\n{table}")
            self.logger.info(tabulate(self.benchmark_results[self.benchmark_results['dataset'] == ds.name][print_order],
                                      headers='keys', tablefmt='psql'))

    # create boxplot diagrams for auc values for each dataset per algorithm
    def create_boxplots_per_algorithm(self, runs, data):
        relevant_results = data[["algorithm", "dataset", "auroc"]]
        for det in self.detectors:
            relevant_results[relevant_results["algorithm"] == det.name].boxplot(by="dataset", figsize=(15, 15))
            plt.title(f"AUC grouped by dataset for {det.name} performing {runs} runs")
            plt.suptitle("")
            plt.tight_layout()
            self.store(plt.gcf(), f"boxplot_auc_for_{det.name}_{runs}_runs")

    # create boxplot diagrams for auc values for each algorithm per dataset
    def create_boxplots_per_dataset(self, runs, data):
        relevant_results = data[["algorithm", "dataset", "auroc"]]
        for ds in self.datasets:
            relevant_results[relevant_results["dataset"] == ds.name].boxplot(by="algorithm", figsize=(15, 15))
            plt.title(f"AUC grouped by algorithm for {ds.name} peforming {runs} runs")
            plt.suptitle("")
            plt.tight_layout()
            self.store(plt.gcf(), f"boxplots_auc_for_{ds.name}_{runs}_runs")

    # create bar charts for averaged pipeline results per algorithm
    def create_bar_charts_per_algorithm(self, runs):
        relevant_results = self.benchmark_results[["algorithm", "dataset", "auroc"]]
        for det in self.detectors:
            relevant_results[relevant_results["algorithm"] == det.name].plot(x="dataset", kind="bar",
                                                                             figsize=(7, 7))
            plt.title(f"AUC for {det.name} performing {runs} runs")
            plt.tight_layout()
            self.store(plt.gcf(), f"barchart_auc_for_{det.name}_{runs}_runs")

    # create bar charts for averaged pipeline results per dataset
    def create_bar_charts_per_dataset(self, runs):
        relevant_results = self.benchmark_results[["algorithm", "dataset", "auroc"]]
        for ds in self.datasets:
            relevant_results[relevant_results["dataset"] == ds.name].plot(x="algorithm", kind="bar", figsize=(7, 7))
            plt.title(f"AUC on {ds.name} performing {runs} runs")
            plt.tight_layout()
            self.store(plt.gcf(), f"barchart_auc_for_{ds.name}_{runs}_runs")

    def store(self, fig, title, extension="pdf"):
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        _dir = self.output_dir if self.output_dir is not None else "reports/figures/"
        path = os.path.join(_dir, f"{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}")
        fig.savefig(path)
        self.logger.info(f"Stored plot at {path}")

    def store_text(self, content, title, extension="txt"):
        timestamp = int(time.time())
        _dir = "reports/tables/"
        path = os.path.join(_dir, f"{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        self.logger.info(f"Stored {extension} file at {path}")

    def generate_latex(self):
        benchmarks = self.benchmarks()
        result = ""
        for ds in self.datasets:
            print_order = ["algorithm", "accuracy", "precision", "recall", "F1-score", "F0.1-score"]
            content = f'''{ds.name}:\n\n{tabulate(benchmarks[benchmarks['dataset'] == ds.name][print_order],
                           headers='keys', tablefmt='latex')}\n\n'''
            result += content
        self.store_text(content=result, title="latex_table_results", extension="tex")
