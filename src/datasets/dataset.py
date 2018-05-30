import abc
import os
import pickle

import pandas as pd


class Dataset:
    """
    ToDo:
        * consider introducing train_data() and test_data() and true_label() or similar
        * think of a useful way to use preprocessed files
    """

    def __init__(self, name: str, processed_path: str):
        self.name = name
        self.processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed/",
                                           processed_path)

        self.data = None

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def load(self):
        """Load data"""

    def data(self) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """Return data, load if necessary"""
        if self.data is None:
            self.load()
        return self.data

    def save(self):
        pickle.dump(self.data, open(self.processed_path, 'wb'))
