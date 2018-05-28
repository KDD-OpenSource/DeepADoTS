import pickle

# For plotting on Windows:
# import matplotlib
# matplotlib.use('TkAgg')

from src.algorithms import DAGMM, LSTM_Enc_Dec
from src.datasets import KDD_Cup
from src.evaluation import get_accuracy_precision_recall_fscore


def main():
    # execute_dagmm()
    execute_lstm_enc_dec()


def execute_dagmm():
    dagmm = DAGMM()
    kdd_cup = KDD_Cup()
    (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print("DAGMM results: ", get_accuracy_precision_recall_fscore(y_test, pred))


def get_synthetic_data():
    with open("data/processed/synthetic", "rb") as f:
        (X_train, y_train, X_test, y_test) = pickle.load(f)
    return (X_train, y_train), (X_test, y_test)


def execute_lstm_enc_dec():
    lstm_enc_dec = LSTM_Enc_Dec(epochs=200, augment_train_data=False, data='lstm_enc_dec_kdd')
    # FIXME: Doesnt print loss/valid loss - not learning
    (X_train, y_train), (X_test, y_test) = KDD_Cup().get_data_dagmm()
    # (X_train, y_train), (X_test, y_test) = get_synthetic_data()
    lstm_enc_dec.fit(X_train, y_train)
    pred = lstm_enc_dec.predict(X_test)
    print("LSTM-Enc_Dec results: ", get_accuracy_precision_recall_fscore(y_test, pred))


if __name__ == '__main__':
    main()
