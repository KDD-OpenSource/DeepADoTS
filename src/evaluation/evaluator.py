class Evaluator:
    def __init__(self, datasets, detectors):
        self.datasets = datasets
        self.detectors = detectors
        self.results = dict()
        
    def evaluate_(self):
        for ds in self.datasets:
            (X_train, y_train, X_test, y_test) = ds.get_data()
            for det in self.detectors:
                det.fit(X_train, y_train)
                score = det.predict(X_test)
                self.results[(type(ds).__name__, type(det).__name__)] = score
    
    def benchmarks(self):          
        df = pd.DataFrame(columns=["name", "accuracy", "precision", "recall", "F1-score"])
        for ds in self.datasets:
            _, _, _, y_test = ds.get_data()
            for det in self.detectors:
                score = self.results[(type(ds).__name__, type(det).__name__)]
                acc, prec, rec, f_score = get_accuracy_precision_recall_fscore(y_test, det.get_binary_label(score))
                df = df.append({"dataset name": type(ds).__name__,
                       "algo name": type(det).__name__,
                       "accuracy": acc,
                       "precision": prec, 
                       "recall": rec, 
                       "F1-score": f_score},
                       ignore_index=True)
        return df
  
    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        return accuracy, precision, recall, f_score