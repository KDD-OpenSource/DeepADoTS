import abc


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self.name = name

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