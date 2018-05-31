import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import progressbar
import matplotlib.pyplot as plt


class Evaluator:
    """
    ToDo:
        * fix roc curve
        * rename benchmarks()
        * refine progressbar or remove (also in requirements)
        * sort columns of df from benchmarks()
    """

    def __init__(self, datasets: list, detectors: list):
        self.datasets = datasets
        self.detectors = detectors
        self.results = dict()

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        fpr, _, _ = roc_curve(y_true, y_pred, pos_label=1)
        return accuracy, precision, recall, f_score, fpr

    def evaluate(self):
        for ds in progressbar.progressbar(self.datasets):
            (X_train, y_train, X_test, y_test) = ds.data()
            for det in progressbar.progressbar(self.detectors):
                det.fit(X_train, y_train)
                score = det.predict(X_test)
                self.results[(ds.name, det.name)] = score

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, det.name)]
                # print(ds, det, score[:20], det.binarize(score[:20]))
                acc, prec, rec, f_score, fpr = self.get_accuracy_precision_recall_fscore(y_test,
                                                                                         det.binarize(score))
                df = df.append({"dataset": ds.name,
                                "approach": det.name,
                                "accuracy": acc,
                                "precision": prec,
                                "recall": rec,
                                "F1-score": f_score,
                                "fpr": fpr[0]},
                               ignore_index=True)
        return df

    def plot_scores(self):
        for ds in self.datasets:
            _, _, X_test, y_test = ds.data()
            subtitle_loc = 'left'
            fig = plt.figure(figsize=(15, 15))
            sp = fig.add_subplot((2*len(self.detectors)+2) * 100 + 11)
            sp.set_title("original test set", loc=subtitle_loc)
            for col in X_test.columns:
                plt.plot(X_test[col])
            sp = fig.add_subplot((2*len(self.detectors)+2) * 100 + 12)
            sp.set_title("binary labels of test set", loc=subtitle_loc)
            plt.plot(y_test)

            subplot_num = 3
            for det in self.detectors:
                sp = fig.add_subplot((2*len(self.detectors)+2) * 100 + 10 + subplot_num)
                sp.set_title("scores of " + det.name, loc=subtitle_loc)
                y_pred = self.results[(ds.name, det.name)]
                plt.plot(np.arange(len(X_test)), [x for x in y_pred])
                threshold_line = len(X_test) * [det.get_threshold(y_pred)]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2*len(self.detectors)+2) * 100 + 10 + subplot_num)
                sp.set_title("binary labels of " + det.name, loc=subtitle_loc)
                plt.plot(np.arange(len(X_test)), [x for x in det.binarize(y_pred)])
                subplot_num += 1
        plt.tight_layout()
        plt.show()

    # TODO: Add more information (algorithms, datasets) in title
    def store(self, fig, title, extension='pdf'):
        timestamp = int(time.time())
        dir = 'reports/figures/'
        path = os.path.join(dir, '{}-{}.{}'.format(title, timestamp, extension))
        fig.savefig(path)
        logging.info('Stored plot at {}'.format(path))

    @staticmethod
    def get_metrics_by_thresholds(y_true: list, y_pred: list, thresholds: list):
        anomalies_by_threshold = np.zeros(len(thresholds))
        acc_by_threshold = np.zeros(len(thresholds))
        prec_by_threshold = np.zeros(len(thresholds))
        recall_by_threshold = np.zeros(len(thresholds))
        f_score_by_threshold = np.zeros(len(thresholds))
        for i in range(len(thresholds)):
            anomaly = np.array(y_pred > thresholds[i], dtype=int)
            anomalies_by_threshold[i] = anomaly.sum()
            acc, prec, recall, f_score, _ = Evaluator.get_accuracy_precision_recall_fscore(y_true, anomaly)
            acc_by_threshold[i] = acc
            prec_by_threshold[i] = prec
            recall_by_threshold[i] = recall
            f_score_by_threshold[i] = f_score
        return anomalies_by_threshold, acc_by_threshold, prec_by_threshold, \
               recall_by_threshold, f_score_by_threshold

    def plot_threshold_comparison(self, steps = 40):
        plots_shape = len(self.detectors), len(self.datasets)
        fig, axes = plt.subplots(*plots_shape, figsize=(15, 15))
        # Ensure two dimensions for iteration
        axes = np.array(axes).reshape(*plots_shape).T
        plt.suptitle('Compare thresholds', fontsize=16)
        for ds, axes_row in zip(self.datasets, axes):
            _, _, X_test, y_test = ds.data()

            for det, ax in zip(self.detectors, axes_row):
                y_pred = np.array(self.results[(ds.name, det.name)])
                if sum(np.isnan(y_pred)) > 0:
                    logging.warn('Prediction contains NaN values. Replacing with 0 for plotting!')
                    y_pred[np.isnan(y_pred)] = 0

                maximum = y_pred.max()
                th = np.linspace(0, maximum, steps)
                anomalies, acc, prec, rec, f_score = self.get_metrics_by_thresholds(
                    y_test, y_pred, th)
                ax.plot(th, anomalies / y_test.shape[0],
                        label=r'anomalies ({} $\rightarrow$ 1)'.format(y_test.shape[0]))
                ax.plot(th, acc, label='accuracy')
                ax.plot(th, prec, label='precision')
                ax.plot(th, rec, label='recall')
                ax.plot(th, f_score, label='f_score')
                ax.set_title('{} on {}'.format(det.name, ds.name))
                ax.set_xlabel('Threshold')
                ax.legend()

        fig.tight_layout()
        # Avoid overlapping title and axis labels
        fig.subplots_adjust(top=0.85, hspace=0.4)
        self.store(fig, 'metrics_by_thresholds')
        plt.show()


    def plot_roc_curves(self):
        # Plot of a ROC curve for all classes

        benchmark_df = self.benchmarks()
        for ds in self.datasets:
            res = benchmark_df[benchmark_df["dataset"] == ds.name]
            plt.figure()
            len_subplot = len(res)
            subplot_count = 1
            for _, line in res.iterrows():
                plt.subplot(len_subplot * 100 + 10 + subplot_count)
                plt.plot(float(line["recall"]), float(line["fpr"]), color='darkorange',
                         lw=2, label='ROC curve (area = %0.2f)' % auc(float(line["recall"]), float(line["fpr"])))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(ds.name)
                plt.legend(loc="lower right")
                subplot_count += 1
            plt.show()
