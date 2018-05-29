import abc


class Algorithm:
    """"
    ToDo:
        * algorithms should have a name
    """

    @abc.abstractmethod
    def fit(self, X, y):
        """
        Train the algorithm on the given dataset
        :param dataset: Wrapper around the raw and processed data
        :return: self
        """

    @abc.abstractmethod
    def predict(self, X):
        """predict"""
