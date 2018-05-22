from src.algorithms.dagmm import DAGMM
from src.algorithms.lstm_enc_dec import LSTM_Enc_Dec
from src.datasets.dataset import KDD_Cup
from src.evaluation import get_accuracy_precision_recall_fscore
from third_party.lstm_enc_dec import preprocess_data


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
    # Load data
    TimeseriesData = preprocess_data.PickleDataLoad(data_type=LSTM_Enc_Dec.args.data, filename=LSTM_Enc_Dec.args.filename,
                                                    augment_test_data=LSTM_Enc_Dec.args.augment)
    train_dataset = TimeseriesData.batchify(LSTM_Enc_Dec.args, TimeseriesData.trainData, LSTM_Enc_Dec.args.batch_size)
    test_dataset = TimeseriesData.batchify(LSTM_Enc_Dec.args, TimeseriesData.testData, LSTM_Enc_Dec.args.eval_batch_size)
    gen_dataset = TimeseriesData.batchify(LSTM_Enc_Dec.args, TimeseriesData.testData, 1)
    lstm_enc_dec = LSTM_Enc_Dec(TimeseriesData, train_dataset, test_dataset, gen_dataset)
    lstm_enc_dec.fit(TimeseriesData, train_dataset, test_dataset, gen_dataset)
    pred = lstm_enc_dec.predict(test_dataset)
    print("LSTM-Enc_Dec results: ", get_accuracy_precision_recall_fscore(TimeseriesData.testLabel, pred))


if __name__ == '__main__':
    main()