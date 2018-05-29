import pickle
import numpy

from src.algorithms import DAGMM, LSTM_Enc_Dec
from src.datasets import KDD_Cup
from src.evaluation import get_accuracy_precision_recall_fscore


def main():
    # execute_dagmm()
    #execute_lstm_enc_dec()
    execute_ensemble_lstm()


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
    # (X_train, y_train), (X_test, y_test) = KDD_Cup().get_data_dagmm()
    (X_train, y_train), (X_test, y_test) = get_synthetic_data()
    lstm_enc_dec = LSTM_Enc_Dec(epochs=20, augment_train_data=True, data='lstm_enc_dec_augmented_2')
    lstm_enc_dec.fit(X_train, y_train)
    pred = lstm_enc_dec.predict(X_test)
    binary_labels = lstm_enc_dec.get_binary_label(pred)
    print("LSTM-Enc_Dec results: ", get_accuracy_precision_recall_fscore(y_test, binary_labels))


# implements an ensemble of lstm_enc_dec
def execute_ensemble_lstm():
    (X_train, y_train), (X_test, y_test) = get_synthetic_data()

    lstm_enc_dec1 = LSTM_Enc_Dec(epochs=1, augment_train_data=True, data='lstm_enc_dec_augmented_1', prediction_window_size=5)
    lstm_enc_dec1.fit(X_train, y_train)
    pred1 = lstm_enc_dec1.predict(X_test)
    binary_labels1 = lstm_enc_dec1.get_binary_label(pred1)
    print("LSTM-Enc_Dec1 results: ", get_accuracy_precision_recall_fscore(y_test, binary_labels1))

    lstm_enc_dec2 = LSTM_Enc_Dec(epochs=1, augment_train_data=True, data='lstm_enc_dec_augmented_2', prediction_window_size=10)
    lstm_enc_dec2.fit(X_train, y_train)
    pred2 = lstm_enc_dec2.predict(X_test)
    binary_labels2 = lstm_enc_dec2.get_binary_label(pred2)
    print("LSTM-Enc_Dec2 results: ", get_accuracy_precision_recall_fscore(y_test, binary_labels2))

    lstm_enc_dec3 = LSTM_Enc_Dec(epochs=1, augment_train_data=True, data='lstm_enc_dec_augmented_3', prediction_window_size=15)
    lstm_enc_dec3.fit(X_train, y_train)
    pred3 = lstm_enc_dec3.predict(X_test)
    binary_labels3 = lstm_enc_dec3.get_binary_label(pred3)
    print("LSTM-Enc_Dec3 results: ", get_accuracy_precision_recall_fscore(y_test, binary_labels3))

    eval_anomaly_scores(pred1, pred2, pred3, y_test)

    '''
    avg = numpy.average((pred1, pred2, pred3), axis=0)
    _min = numpy.min((pred1, pred2, pred3), axis=0)
    _max = numpy.average((pred1, pred2, pred3), axis=0)
    print("LSTM-Enc_Dec combined avg results: ", get_accuracy_precision_recall_fscore(y_test, numpy.round(avg)))
    print("LSTM-Enc_Dec combined min results: ", get_accuracy_precision_recall_fscore(y_test, numpy.round(_min)))
    print("LSTM-Enc_Dec combined max results: ", get_accuracy_precision_recall_fscore(y_test, numpy.round(_max)))
    '''


def eval_anomaly_scores(anomaly_scores1, anomaly_scores2, anomaly_scores3, ground_truth):
    avg = numpy.average((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
    _min = numpy.min((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)
    _max = numpy.average((anomaly_scores1, anomaly_scores2, anomaly_scores3), axis=0)

    lstm_enc_dec = LSTM_Enc_Dec(epochs=1, augment_train_data=True, data='lstm_enc_dec_augmented')

    binary_labels1 = lstm_enc_dec.get_binary_label(avg)
    binary_labels2 = lstm_enc_dec.get_binary_label(_min)
    binary_labels3 = lstm_enc_dec.get_binary_label(_max)

    print("LSTM-Enc_Dec combined avg results: ", get_accuracy_precision_recall_fscore(ground_truth, binary_labels1))
    print("LSTM-Enc_Dec combined min results: ", get_accuracy_precision_recall_fscore(ground_truth, binary_labels2))
    print("LSTM-Enc_Dec combined max results: ", get_accuracy_precision_recall_fscore(ground_truth, binary_labels3))


if __name__ == '__main__':
    main()
