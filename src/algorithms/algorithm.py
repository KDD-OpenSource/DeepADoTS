import numpy as np


class Algorithm:
    def fit(self, X, y):
        """
        Train the algorithm on the given dataset
        :param dataset: Wrapper around the raw and processed data
        :return: self
        """
        raise NotImplementedError

    def predict(self, X: np.array):
        raise NotImplementedError
