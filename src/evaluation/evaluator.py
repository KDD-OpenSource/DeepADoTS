import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support as prf, accuracy_score

class Evaluator:
    def __init__(self, datasets, detectors):
        self.datasets = datasets
        self.detectors = detectors
        self.results = dict()
        self.benchmark_df = pd.DataFrame()
        
    def evaluate(self):
        for ds in self.datasets:
            (X_train, y_train, X_test, y_test) = ds.get_data()
            for det in self.detectors:
                det.fit(X_train, y_train)
                score = det.predict(X_test)
                self.results[(type(ds).__name__, type(det).__name__)] = score
    
    def benchmarks(self):          
        df = pd.DataFrame(columns=["dataset", "approach", "accuracy", "precision", "recall", "F1-score", "fpr"])
        for ds in self.datasets:
            _, _, _, y_test = ds.get_data()
            for det in self.detectors:
                score = self.results[(type(ds).__name__, type(det).__name__)]
                acc, prec, rec, f_score, fpr = self.get_accuracy_precision_recall_fscore(y_test, det.get_binary_label(score))
                df = df.append({"dataset": type(ds).__name__,
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
            _, _, X_test, y_test = ds.get_data()
            plt.subplot((2*len(self.detectors)+2) * 100 + 11)
            for col in X_test.columns:
                plt.plot(X_test[col])
            plt.subplot((2*len(self.detectors)+2) * 100 + 12)
            plt.plot(y_test)
            
            subplot_num = 3
            for det in self.detectors:
                plt.subplot((2*len(self.detectors)+2) * 100 + 10 + subplot_num)
                y_pred = self.results[(type(ds).__name__, type(det).__name__)]
                plt.plot(np.arange(len(X_test)), [x for x in y_pred])
                threshold_line = len(X_test) * [det.get_threshold(y_pred)]
                plt.plot([x for x in threshold_line])
                subplot_num += 1
                
                plt.subplot((2*len(self.detectors)+2) * 100 + 10 + subplot_num)
                plt.plot(np.arange(len(X_test)), [x for x in det.get_binary_label(y_pred)])
                subplot_num += 1
        plt.legend()
        plt.show()
        
        self.plot_roc_curves()
        
    def plot_roc_curves(self):
        #Plot of a ROC curve for all classes
        for ds in self.datasets:
            res = self.benchmark_df[self.benchmark_df["dataset"] == type(ds).__name__]
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
                plt.title(type(ds).__name__)
                plt.legend(loc="lower right")
                subplot_count+=1
            plt.show()
  
    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        fpr, _, _ = roc_curve(y_true, y_pred, pos_label=1)
        return accuracy, precision, recall, f_score, fpr