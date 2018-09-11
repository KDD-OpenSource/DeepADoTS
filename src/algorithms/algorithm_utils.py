import abc
import logging
import random

import numpy as np
import torch
import tensorflow as tf
from tensorflow.python.client import device_lib
from torch.autograd import Variable


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, module_name, name, seed, details=False):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.details = details
        self.prediction_details = {}

        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return anomaly score
        """


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
