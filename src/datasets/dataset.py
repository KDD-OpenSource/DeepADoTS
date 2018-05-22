import os

import numpy as np


class Dataset:
    def __init__(self, name, raw_path, processed_path):
        self.name = name
        self.raw_path = os.path.join(os.getcwd(), "data/raw/", raw_path)
        self.processed_path = os.path.join(os.getcwd(), "data/processed/", processed_path)


class KDD_Cup(Dataset):
    def __init__(self):
        super(KDD_Cup, self).__init__("KDD Cup '99", "kddcup-data_10_percent_corrected.txt", "kdd_cup.npz")

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
        labels = data["kdd"][:, -1]
        features = data["kdd"][:, :-1]

        normal_data = features[labels == 1]
        normal_labels = labels[labels == 1]

        attack_data = features[labels == 0]
        attack_labels = labels[labels == 0]

        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        N_train = N_attack // 2
        train = attack_data[randIdx[:N_train]]
        train_labels = attack_labels[randIdx[:N_train]]

        test = attack_data[randIdx[N_train:]]
        test_labels = attack_labels[randIdx[N_train:]]

        test = np.concatenate((test, normal_data), axis=0)
        test_labels = np.concatenate((test_labels, normal_labels), axis=0)

        return (train, train_labels), (test, test_labels)