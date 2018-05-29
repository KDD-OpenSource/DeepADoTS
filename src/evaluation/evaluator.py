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
        * refactor (type(det).__name__ into name attribute
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
    def get_accuracy_precision_recall_fscore(y_true, y_pred):
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
                self.results[(ds.name, type(det).__name__)] = score

    def benchmarks(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                score = self.results[(ds.name, type(det).__name__)]
                acc, prec, rec, f_score, fpr = self.get_accuracy_precision_recall_fscore(y_test,
                                                                                         det.get_binary_label(score))
                df = df.append({"dataset": 
                                ds.name,
                                "approach": type(det).__name__,
                                "accuracy": acc,
                                "precision": prec,
                                "recall": rec,
                                "F1-score": f_score,
                                "fpr": fpr[0]},
                               ignore_index=True)
        self.benchmark_df = df
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
                sp.set_title("scores of " + type(det).__name__, loc=subtitle_loc)
                y_pred = self.results[(ds.name, type(det).__name__)]
                plt.plot(np.arange(len(X_test)), [x for x in y_pred])
                threshold_line = len(X_test) * [det.get_threshold(y_pred)]
                plt.plot([x for x in threshold_line])
                subplot_num += 1

                sp = fig.add_subplot((2*len(self.detectors)+2) * 100 + 10 + subplot_num)
                sp.set_title("binary labels of " + type(det).__name__, loc=subtitle_loc)
                plt.plot(np.arange(len(X_test)), [x for x in det.get_binary_label(y_pred)])
                subplot_num += 1
        plt.legend()
        plt.tight_layout()
        plt.show()
        # self.plot_roc_curves()

    def plot_roc_curves(self):
        # Plot of a ROC curve for all classes
        for ds in self.datasets:
            res = self.benchmark_df[self.benchmark_df["dataset"] == ds.name]
            plt.figure()
            len_subplot = len(res)
            subplot_count = 1
            print(res)
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
