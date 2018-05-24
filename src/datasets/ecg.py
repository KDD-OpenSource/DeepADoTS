"""
    Paper: https://arxiv.org/pdf/1607.00148.pdf
    Download source: http://www.cs.ucr.edu/~eamonn/discords/
"""

import numpy as np

from third_party.lstm_enc_dec import train_predictor
from third_party.lstm_enc_dec import preprocess_data

from .dataset import Dataset


class ECG(Dataset):
    def __init__(self, pickle_path="ecg/whole/chfdb_chf13_45590.pkl"):
        # raw path is not used: "ecg/chfdb_chf13_45590.txt"
        super(ECG, self).__init__("ECG", "", pickle_path)
        self.args = train_predictor.get_args()
        self.augment_test_data = True
        is_ecg = (pickle_path == "ecg/whole/chfdb_chf13_45590.pkl")
        self.trainTimeseriesData = preprocess_data.PickleDataLoad(
            data_type='ECG', filename=self.processed_path, augment_test_data=self.augment_test_data,
            ecg=is_ecg
        )
        self.testTimeseriesData = preprocess_data.PickleDataLoad(
            data_type='ECG', filename=self.processed_path, augment_test_data=False,
            ecg=is_ecg
        )

    def get_device(self):
        return self.args.device

    def get_feature_dim(self):
        return self.trainTimeseriesData.trainData.size(1)

    def get_test_labels(self):
        return self.testTimeseriesData.testLabel.to(self.args.device)

    def get_test_data(self):
        return self.testTimeseriesData

    def get_train_data(self):
        return self.trainTimeseriesData
