import abc
import copy
import logging


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, module_name, name, framework):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.framework = framework

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def fit(self, X, y):
        """
        Train the algorithm on the given dataset
        :param X:
        :param y:
        :return: self
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return score
        """

    @abc.abstractmethod
    def binarize(self, score, threshold=None):
        """
        :param threshold:
        :param score
        :return binary_labels
        """

    @abc.abstractmethod
    def threshold(self, score):
        """
        :param score
        :return threshold:
        """

    @abc.abstractmethod
    def set_seed(self, seed):
        """
        :param seed:
        :return:
        """

    class Frameworks:
        PyTorch, Tensorflow = range(2)

    def clone(self):
        return copy.deepcopy(self)
