import abc
import os
import pandas as pd


class Dataset:
    """
    ToDo:
        * consider introducing train_data() and test_data() and true_label() or similar
        * think of a useful way to use preprocessed files
    """

    def __init__(self, name: str, raw_path: str, processed_path: str):
        self.name = name
        self.raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/", raw_path)
        self.processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed/",
                                           processed_path)

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def data(self) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """returns data"""
