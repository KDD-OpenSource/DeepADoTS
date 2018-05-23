from src.algorithms import DAGMM, LSTM_Enc_Dec
from src.datasets import KDD_Cup, ECG
from src.evaluation import get_accuracy_precision_recall_fscore


def main():
    #execute_dagmm()
    execute_lstm_enc_dec()


def execute_dagmm():
    dagmm = DAGMM()
    kdd_cup = KDD_Cup()
    (X_train, y_train), (X_test, y_test) = kdd_cup.get_data_dagmm()
    dagmm.fit(X_train, y_train)
    pred = dagmm.predict(X_test)
    print("DAGMM results: ", get_accuracy_precision_recall_fscore(y_test, pred))


def execute_lstm_enc_dec():
    args = train_predictor.get_args()
    # Load data
    ecg = ECG()
    lstm_enc_dec = LSTM_Enc_Dec(*ecg.get_lstm_enc_dec_data())
    lstm_enc_dec.fit()
    pred = lstm_enc_dec.predict()
    TimeseriesData = self.TimeseriesData
    print("pred: ", pred)
    print("test_label: ", TimeseriesData.testLabel.to(args.device))
    print("LSTM-Enc_Dec results: ", get_accuracy_precision_recall_fscore(TimeseriesData.testLabel.to(args.device), pred))


if __name__ == '__main__':
    main()
