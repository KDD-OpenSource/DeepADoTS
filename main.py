from src.algorithms.dagmm import DAGMM
from src.datasets.dataset import KDD_Cup
from src.evaluation import get_accuracy_precision_recall_fscore


def main():
    dagmm = DAGMM()
    kdd_cup = KDD_Cup()
    (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print(get_accuracy_precision_recall_fscore(y_test, pred))


if __name__ == '__main__':
    main()
