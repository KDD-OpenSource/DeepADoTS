"""
    Paper: https://arxiv.org/pdf/1607.00148.pdf
    Download source: http://www.cs.ucr.edu/~eamonn/discords/
"""

import numpy as np

from third_party.lstm_enc_dec import preprocess_data

from .dataset import Dataset


class ECG(Dataset):
    def __init__(self):
        super(ECG, self).__init__("ECG", "", "chfdb_chf13_45590.pkl")
        self.batch_size = 64
        self.eval_batch_size = 64
        self.augment_test_data = True

    def get_lstm_enc_dec_data(self):
        TimeseriesData = preprocess_data.PickleDataLoad(
            data_type='ECG', filename=self.processed_path, augment_test_data=self.augment_test_data
        )
        train_dataset = TimeseriesData.batchify(args, TimeseriesData.trainData, self.batch_size)
        test_dataset = TimeseriesData.batchify(args, TimeseriesData.testData, self.eval_batch_size)
        gen_dataset = TimeseriesData.batchify(args, TimeseriesData.testData, 1)

        return (TimeseriesData, train_dataset, test_dataset, gen_dataset)
