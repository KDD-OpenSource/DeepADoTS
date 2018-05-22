from src.algorithms.dagmm import DAGMM
from src.algorithms.lstm_enc_dec import LSTM_Enc_Dec
from third_party.lstm_enc_dec import train_predictor
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
    args = train_predictor.get_args()
    # Load data
    TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename,
                                                    augment_test_data=args.augment)
    train_dataset = TimeseriesData.batchify(args, TimeseriesData.trainData, args.batch_size)
    eval_dataset = TimeseriesData.batchify(args, TimeseriesData.testData, args.eval_batch_size)
    gen_dataset = TimeseriesData.batchify(args, TimeseriesData.testData, 1)
    lstm_enc_dec = LSTM_Enc_Dec(TimeseriesData, train_dataset, eval_dataset, gen_dataset)
    lstm_enc_dec.fit()

    TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename,
                                                    augment_test_data=False)
    train_dataset = TimeseriesData.batchify(args, TimeseriesData.trainData[:TimeseriesData.length], bsz=1)
    test_dataset = TimeseriesData.batchify(args, TimeseriesData.testData, bsz=1)
    pred = lstm_enc_dec.predict(TimeseriesData, train_dataset, test_dataset)
    print("pred: ", pred)
    print("test_label: ", TimeseriesData.testLabel.to(args.device))
    print("LSTM-Enc_Dec results: ", get_accuracy_precision_recall_fscore(TimeseriesData.testLabel.to(args.device), pred))


if __name__ == '__main__':
    main()