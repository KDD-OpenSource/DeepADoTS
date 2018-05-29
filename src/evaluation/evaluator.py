import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import accuracy_score
import progressbar


class Evaluator:
    """
    ToDo:
        * refactor (type(det).__name__ into name attribute
        * include plots
        * rename benchmarks()
        * move _binary_label() into detectors
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
        return accuracy, precision, recall, f_score

    @staticmethod
    def _binary_label(score):
        threshold = np.mean(score) + np.std(score)
        return np.where(score >= threshold, 1, 0)

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
                acc, prec, rec, f_score = self.get_accuracy_precision_recall_fscore(y_test, self._binary_label(score))
                df = df.append({"dataset_name": ds.name,
                                "algorithm_mame": type(det).__name__,
                                "accuracy": acc,
                                "precision": prec,
                                "recall": rec,
                                "f1-score": f_score},
                               ignore_index=True)
        return df

    def plot_roc(self):
        """ Please, return a figure. """
        raise NotImplementedError
