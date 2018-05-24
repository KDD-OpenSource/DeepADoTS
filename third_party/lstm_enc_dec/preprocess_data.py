import os
import torch
from torch import device
import glob
import datetime
import numpy as np
import shutil
from pathlib import Path
import pickle


def normalization(seqData, max, min):
    return (seqData - min) / (max - min)


def standardization(seqData, mean, std):
    return (seqData - mean) / std


def reconstruct(seqData, mean, std):
    return seqData * std + mean


class PickleDataLoad(object):
    # augment=True: Same shape for test and train
    # for ecg: define filename
    # for other datasets (passed as pandas or numpy objects): define input_data
    def __init__(self, data_type="ECG", filename="", input_data=None, augment_test_data=False):
        # FIXME: Not supported because shapes do not match in line 46 (expand_as)
        self.augment_test_data = augment_test_data
        if input_data is not None:
            assert len(input_data) == 4, 'Input should be train and test (X and y)'
            self.trainData, self.trainLabel = self.preprocessing(input_data, train=True)
            self.testData, self.testLabel = self.preprocessing(input_data, train=False)
        else:
            assert len(filename) > 0
            self.trainData, self.trainLabel = self.preprocessing_ecg(Path('data', 'processed', data_type, 'train', filename),
                                                                     train=True)
            self.testData, self.testLabel = self.preprocessing_ecg(Path('data', 'processed', data_type, 'test', filename),
                                                                   train=False)

    def augmentation(self, data, label, noise_ratio=0.05, noise_interval=0.0005, max_length=100000):
        noiseSeq = torch.randn(data.size())
        augmentedData = data.clone()
        augmentedLabel = label.clone()
        for i in np.arange(0, noise_ratio, noise_interval):
            scaled_noiseSeq = noise_ratio * self.std.expand_as(data) * noiseSeq
            augmentedData = torch.cat([augmentedData, data + scaled_noiseSeq], dim=0)
            augmentedLabel = torch.cat([augmentedLabel, label])
            if len(augmentedData) > max_length:
                augmentedData = augmentedData[:max_length]
                augmentedLabel = augmentedLabel[:max_length]
                break

        return augmentedData, augmentedLabel

    def preprocessing(self, input_data, train=True, augmentation=True):
        """ Read, Standardize, Augment """
        (X_train, y_train, X_test, y_test) = input_data

        if train:
            label = torch.FloatTensor(np.array(y_train))
            data = torch.FloatTensor(np.array(X_train))
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            self.length = len(data)
            if augmentation:
                data, label = self.augmentation(data, label)
        else:
            label = torch.FloatTensor(np.array(y_test))
            data = torch.FloatTensor(np.array(X_test))
            if self.augment_test_data:
                data, label = self.augmentation(data, label)

        data = standardization(data, self.mean, self.std)

        return data, label

    def preprocessing_ecg(self, path, train=True):
        """ Read, Standardize, Augment """

        with open(str(path), 'rb') as f:
            pickled_data = pickle.load(f)
            data = torch.FloatTensor(pickled_data)
            label = data[:, -1]
            data = data[:, :-1]
        if train:
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            self.length = len(data)
            data, label = self.augmentation(data, label)
        else:
            if self.augment_test_data:
                data, label = self.augmentation(data, label)

        data = standardization(data, self.mean, self.std)

        return data, label

    def batchify(self, args, data, bsz):
        nbatch = data.size(0) // bsz
        trimmed_data = data.narrow(0, 0, nbatch * bsz)
        batched_data = trimmed_data.contiguous().view(bsz, -1, trimmed_data.size(-1)).transpose(0, 1)

        batched_data = batched_data.to(device(args.device))
        return batched_data
