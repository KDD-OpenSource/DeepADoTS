import numpy as np
import pandas as pd
from agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator

from src.datasets.dataset import Dataset


class SyntheticDataGenerator:
    @staticmethod
    def get_extreme1():
        name = "extreme1"
        path = "empty"  # ToDo
        length = 1500
        n = 5
        k = 3
        random_state = 1337
        outlier_config = {
        }
        return SyntheticData(name, path, length, n, k, outlier_config, random_state)

    @staticmethod
    def get_schwifty():
        pass


class SyntheticData(Dataset):
    """

    ToDo:
        * refactor data()
        * much more
    """

    def __init__(self, name: str, path: str, length: int=1000, n: int=4, k: int=2, config: dict=None,
                 random_state: int=None):
        super(SyntheticData, self).__init__(name, "", path)
        self.length = length
        self.n = n
        self.k = k
        self.config = config if config is not None else {}
        if random_state is not None:
            np.random.seed(random_state)

    def data(self):
        dg = MultivariateDataGenerator(self.length, self.n, self.k, shift_config={1: 25, 2: 20})
        X_test = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)
        X_test['y'] = np.zeros(self.length)
        X_train = X_test.copy()

        for timeseries in range(self.n):
            num_outliers = 10
            outlier_pos = sorted(np.random.choice(range(self.length), num_outliers, replace=False))

            timestamps = []
            for outlier in outlier_pos:
                timestamps.append((outlier,))

            X_test = dg.add_outliers(self.config)
            X_test['y'] = np.where(X_test.index.isin(outlier_pos), 1, 0)

        y_train = X_train['y']
        y_test = X_train['y']
        del X_train['y']
        del X_test['y']

        return X_train, y_train, X_test, y_test

    @staticmethod
    def add_missing_values(X_train: pd.DataFrame, missing_percentage: float, per_column: bool=True) -> pd.DataFrame:
        if per_column:
            for col in X_train.columns:
                missing_idxs = np.random.choice(len(X_train), int(missing_percentage*len(X_train)), replace=False)
                X_train[col][missing_idxs] = np.nan
        else:
            missing_idxs = np.random.choice(len(X_train), int(missing_percentage*len(X_train)), replace=False)
            X_train.iloc[missing_idxs] = [np.nan] * X_train.shape[-1]
        return X_train
