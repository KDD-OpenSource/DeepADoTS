import logging
import os
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

from .config import init_logging


class Evaluator:
    def __init__(self, datasets: list, detectors: list):
        self.datasets = datasets
        self.detectors = detectors
        self.results = dict()
        init_logging()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, _ = prf(y_true, y_pred, average="binary")
        f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.1)
        return accuracy, precision, recall, f_score, f01_score

    @staticmethod
    def get_auroc(det, ds, score):
        _, _, _, y_test = ds.data()
        y_pred = det.binarize(score)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        return auc(fpr, tpr)

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
                    self.logger.error(f"An exception occured while training {det.name} on {ds}: {e}")
                    self.logger.error(traceback.format_exc())
                    self.results[(ds.name, det.name)] = np.zeros_like(y_test)

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                y_pred = det.binarize(score)
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

    def store(self, fig, title, extension="pdf"):
        timestamp = int(time.time())
        dir = "reports/figures/"
        path = os.path.join(dir, f"{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}")
        fig.savefig(path)
        self.logger.info(f"Stored plot at {path}")

    @staticmethod
    def get_metrics_by_thresholds(det, y_true: list, y_pred: list, thresholds: list):
        for threshold in thresholds:
            anomaly = det.binarize(y_pred, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_true, anomaly)
            yield (anomaly.sum(), *metrics)

    def plot_scores(self, store=True):
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
        plots_shape = len(self.detectors), len(self.datasets)
        fig, axes = plt.subplots(*plots_shape, figsize=(15, 15))
        # Ensure two dimensions for iteration
        axes = np.array(axes).reshape(*plots_shape).T
        plt.suptitle("Compare thresholds", fontsize=16)
        for ds, axes_row in zip(self.datasets, axes):
            _, _, X_test, y_test = ds.data()

            for det, ax in zip(self.detectors, axes_row):
                y_pred = np.array(self.results[(ds.name, det.name)])
                if np.isnan(y_pred).any():
                    self.logger.warning("Prediction contains NaN values. Replacing with 0 for plotting!")
                    y_pred[np.isnan(y_pred)] = 0

                maximum = y_pred.max()
                th = np.linspace(0, maximum, steps)
                metrics = list(self.get_metrics_by_thresholds(det, y_test, y_pred, th))
                metrics = np.array(metrics).T
                anomalies, _, prec, rec, f_score, f01_score = metrics

                ax.plot(th, anomalies / len(y_test),
                        label=fr"anomalies ({len(y_test)} $\rightarrow$ 1)")
                ax.plot(th, prec, label="precision")
                ax.plot(th, rec, label="recall")
                ax.plot(th, f_score, label="f_score")
                ax.plot(th, f01_score, label="f01_score")
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
                y_pred = det.binarize(score)
                fpr, tpr, _ = roc_curve(y_test, y_pred)
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
                plt.title(det.name)
                plt.legend(loc="lower right")
            plt.tight_layout()
            if store:
                self.store(fig, f"roc_{ds.name}")
            figures.append(fig)
        return figures

    def plot_auroc(self, store=True, title='AUROC'):
        benchmarks = self.benchmarks()
        dataset_names = [ds.name for ds in self.datasets]
        fig = plt.figure(figsize=(7, 7))
        for det in self.detectors:
            aurocs = benchmarks[benchmarks['algorithm'] == det.name]['auroc']
            plt.plot(aurocs.values, label=det.name)
        plt.xticks(range(len(self.datasets)), dataset_names, rotation=90)
        plt.legend()
        plt.xlabel('Dataset')
        plt.ylabel('Area under Receiver Operating Characteristic')
        plt.title(title)
        fig.tight_layout()
        if store:
            self.store(fig, f"auroc")
        return fig

    def print_tables(self):
        benchmarks = self.benchmarks()
        for ds in self.datasets:
            self.logger.info(f"Dataset: {ds.name}")
            print_order = ["algorithm", "accuracy", "precision", "recall", "F1-score", "F0.1-score"]
            self.logger.info(tabulate(benchmarks[benchmarks['dataset'] == ds.name][print_order],
                                      headers='keys', tablefmt='psql'))
