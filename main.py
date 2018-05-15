from src.dataset import Dataset
from src.algorithms.dagmm import DAGMM
from src.data_loader import get_train_test_split
from src.evaluation import get_accuracy_precision_recall_fscore


kdd_cup = Dataset("KDD Cup '99", "kddcup-data_10_percent_corrected.txt", "kdd_cup.npz")
dagmm = DAGMM(data_path=kdd_cup.processed_path, num_epochs=1)
X_train, X_test, y_train, y_test = get_train_test_split(kdd_cup.processed_path)
dagmm.fit(X_train, y_train)
gt, pred = dagmm.predict(X_test)
print(get_accuracy_precision_recall_fscore(gt, pred))