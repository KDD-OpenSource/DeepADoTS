import abc
import logging
import random
import sys

import numpy as np
import torch
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from tensorflow.python.client import device_lib
from torch.autograd import Variable


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, module_name, name, seed):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X, X_test=None, y_test=None):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return anomaly score
        """

    def epoch_eval(self, X_test, y_test):
        score = self.predict(X_test.copy())
        if np.isnan(score).all():
            score = np.zeros_like(score)
        score_nonan = score.copy()
        # Rank NaN below every other value in terms of anomaly score
        score_nonan[np.isnan(score_nonan)] = np.nanmin(score_nonan) - sys.float_info.epsilon
        fpr, tpr, _ = roc_curve(y_test, score_nonan)
        return auc(fpr, tpr)


class PyTorchUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.framework = 0

    @property
    def device(self):
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.gpu is not None else 'cpu')

    def to_var(self, t, **kwargs):
        # ToDo: check whether cuda Variable.
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(self.device)


class TensorflowUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            tf.set_random_seed(seed)
        self.framework = 1

    @property
    def device(self):
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        return tf.device(gpus[self.gpu] if gpus and self.gpu is not None else '/cpu:0')
