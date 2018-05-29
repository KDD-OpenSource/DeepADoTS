class Algorithm:
    def fit(self, X, y):
        """
        Train the algorithm on the given dataset
        :param dataset: Wrapper around the raw and processed data
        :return: self
        """
        raise NotImplementedError

    def predict(self, X):
        """
        :return scores
        """
        raise NotImplementedError
        
    def get_binary_label(y):
        """
        :param scores
        :return binary_labels
        """
        raise NotImplementedError
