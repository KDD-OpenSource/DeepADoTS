"""
    Paper: https://arxiv.org/pdf/1607.00148.pdf
    Download source: http://www.cs.ucr.edu/~eamonn/discords/
"""

import numpy as np

from third_party.lstm_enc_dec import train_predictor
from third_party.lstm_enc_dec import preprocess_data

from .dataset import Dataset


class ECG(Dataset):
    def __init__(self):
        super(ECG, self).__init__("ECG", "ecg/chfdb_chf13_45590.txt", "ecg/whole/chfdb_chf13_45590.pkl")
        self.args = train_predictor.get_args()
        self.batch_size = 64
        self.eval_batch_size = 64
        self.augment_test_data = True
        self.TimeseriesData = preprocess_data.PickleDataLoad(
            data_type='ECG', filename=self.processed_path, augment_test_data=self.augment_test_data
        )

    def get_device():
        return self.args.device

    def get_test_labels():
        return self.TimeseriesData.testLabel.to(self.args.device)

    def get_lstm_enc_dec_data(self):

        train_dataset = self.TimeseriesData.batchify(self.args, self.TimeseriesData.trainData, self.batch_size)
        test_dataset = self.TimeseriesData.batchify(self.args, self.TimeseriesData.testData, self.eval_batch_size)
        gen_dataset = self.TimeseriesData.batchify(self.args, self.TimeseriesData.testData, 1)

        return (self.TimeseriesData, train_dataset, test_dataset, gen_dataset)
