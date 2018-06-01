import logging
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import progressbar
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import roc_curve, auc


class Evaluator:
    def __init__(self, datasets: list, detectors: list):
        self.datasets = datasets
        self.detectors = detectors
        self.results = dict()

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.1)
        return accuracy, precision, recall, f_score, f01_score

    def evaluate(self):
        for ds in progressbar.progressbar(self.datasets):
            (X_train, y_train, X_test, y_test) = ds.data()
            for det in progressbar.progressbar(self.detectors):
                print("Training " + det.name + " on " + str(ds))
                try:
                    det.fit(X_train, y_train)
                    score = det.predict(X_test)
                    self.results[(ds.name, det.name)] = score
                except Exception as e:
                    print(f"ERROR: An error occured while training {det.name} on {ds}: {e}")
                    self.results[(ds.name, det.name)] = np.zeros_like(y_test)

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                y_pred = det.binarize(score)
                acc, prec, rec, f1_score, f01_score = self.get_accuracy_precision_recall_fscore(y_test, y_pred)
                df = df.append({"dataset": ds.name,
                                "algorithm": det.name,
                                "accuracy": acc,
                                "precision": prec,
                                "recall": rec,
                                "F1-score": f1_score,
                                "F0.1-score": f01_score},
                               ignore_index=True)
        return df

    def store(self, fig, title, extension="pdf"):
        # ToDo: Add more information (algorithms, datasets) in title
        timestamp = int(time.time())
        dir = "reports/figures/"
        path = os.path.join(dir, f"{title}-{len(self.detectors)}-{len(self.datasets)}-{timestamp}.{extension}")
        fig.savefig(path)
        logging.info(f"Stored plot at {path}")

    @staticmethod
    def get_metrics_by_thresholds(det, y_true: list, y_pred: list, thresholds: list):
        for threshold in thresholds:
            anomaly = det.binarize(y_pred, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_true, anomaly)
            yield (anomaly.sum(), *metrics)

    def plot_scores(self, store=True):
        figures = []
        for ds in self.datasets:
            _, _, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            sp = fig.add_subplot((2 * len(self.detectors) + 2), 1, 1)
            sp.set_title("original test set", loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])
            sp = fig.add_subplot((2 * len(self.detectors) + 2), 1, 2)
            sp.set_title("binary labels of test set", loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 3
            for det in self.detectors:
                sp = fig.add_subplot((2 * len(self.detectors) + 2), 1, subplot_num)
                sp.set_title(f"scores of {det.name}", loc=subtitle_loc)
                score = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(score)), [x for x in score])
                threshold_line = len(score) * [det.get_threshold(score)]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2 * len(self.detectors) + 2), 1, subplot_num)
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
                    logging.warning("Prediction contains NaN values. Replacing with 0 for plotting!")
                    y_pred[np.isnan(y_pred)] = 0

                maximum = y_pred.max()
                th = np.linspace(0, maximum, steps)
                metrics = list(self.get_metrics_by_thresholds(det, y_test, y_pred, th))
                metrics = np.array(metrics).T
                anomalies, acc, prec, rec, f_score, f01_score = metrics

                ax.plot(th, anomalies / len(y_test),
                        label=fr"anomalies ({len(y_test)} $\rightarrow$ 1)")
                ax.plot(th, acc, label="accuracy")
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
            fig = plt.figure(figsize=(fig_scale*len(self.detectors), fig_scale))
            fig.suptitle(f"ROC curve on {ds.name}", fontsize=14, y="1.1")
            subplot_count = 1
            for det in self.detectors:
                logging.info(f"Plotting {det.name} on {ds.name}")
                score = self.results[(ds.name, det.name)]
                y_pred = det.binarize(score)
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                plt.subplot(1, len(self.detectors), subplot_count)
                plt.plot(fpr, tpr, color="darkorange",
                         lw=2, label='area = %0.2f' % roc_auc)
                subplot_count += 1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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

    def print_tables(self):
        benchmarks = self.benchmarks()
        for ds in self.datasets:
            print(f"Dataset: {ds.name}")
            print_order = ["algorithm", "accuracy", "precision", "recall", "F1-score", "F0.1-score"]
            print(tabulate(benchmarks[benchmarks['dataset'] == ds.name][print_order],
                  headers='keys', tablefmt='psql'))
