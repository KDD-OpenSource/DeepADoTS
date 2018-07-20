import numpy as np
import pandas as pd

from .real_datasets import RealDataset


class KDDCup(RealDataset):
    def __init__(self, seed):
        super().__init__(
            name="KDD Cup '99", raw_path='kddcup-data_10_percent_corrected.txt', file_name='kdd_cup.npz'
        )
        self.seed = seed

    def load(self):
        (a, b), (c, d) = self.get_data_dagmm()
        self._data = (a, b, c, d)

    def get_data_dagmm(self):
        """
        This approach is used by the DAGMM paper (Zong et al., 2018) and was first described in Zhai et al.,
        Deep structured energy based models for anomaly detection:
        "As 20% of data samples are labeled as “normal” and the rest are labeled as “attack”, “normal” samples are in a
        minority group; therefore, “normal” ones are treated as anomalies in this task" - Zong et al., 2018
        "[...]in each run, we take 50% of data by random sampling for training with the rest 50% reserved for testing,
        and only data samples from the normal class are used for training models.[...] - Zong et al., 2018"
        :return: (X_train, y_train), (X_test, y_test)
        """
        data = np.load(self.processed_path)
        np.random.seed(self.seed)

        labels = data['kdd'][:, -1]
        features = data['kdd'][:, :-1]

        normal_data = features[labels == 1]
        normal_labels = labels[labels == 1]

        attack_data = features[labels == 0]
        attack_labels = labels[labels == 0]

        n_attack = attack_data.shape[0]

        rand_idx = np.arange(n_attack)
        np.random.shuffle(rand_idx)
        n_train = n_attack // 2

        train = attack_data[rand_idx[:n_train]]
        train_labels = attack_labels[rand_idx[:n_train]]

        test = attack_data[rand_idx[n_train:]]
        test_labels = attack_labels[rand_idx[n_train:]]

        test = np.concatenate((test, normal_data), axis=0)
        test_labels = np.concatenate((test_labels, normal_labels), axis=0)

        return (pd.DataFrame(data=train), pd.DataFrame(data=train_labels)), (
            pd.DataFrame(data=test), pd.DataFrame(data=test_labels))
