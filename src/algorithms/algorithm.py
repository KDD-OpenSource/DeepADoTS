import abc


class Algorithm:
    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def fit(self, X, y):
        """
        Train the algorithm on the given dataset
        :param dataset: Wrapper around the raw and processed data
        :return: self
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return scores
        """

    @abc.abstractmethod
    def get_binary_label(self, y):
        """
        :param scores
        :return binary_labels
        """
