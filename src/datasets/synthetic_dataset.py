import numpy as np
import pandas as pd
from agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator

from . import Dataset


class SyntheticDataset(Dataset):
    """

    ToDo:
        * refactor data()
        * much more
    """

    def __init__(self, length: int=1000, n: int=4, k: int=2, config: dict={},
                 shift_config: dict={}, random_state: int=None,
                 pollution_config: dict={}, **kwargs):
        super().__init__(kwargs)

        self.length = length
        self.n = n
        self.k = k
        self.config = config
        self.shift_config = shift_config
        self.pollution_config = pollution_config
        if random_state is not None:
            np.random.seed(random_state)

    def load(self):
        generator = MultivariateDataGenerator(self.length, self.n, self.k, shift_config=self.shift_config)
        generator.generate_baseline(initial_value_min=-4, initial_value_max=4)

        X_train = generator.add_outliers(self.pollution_config)
        y_train = self._label_outliers(self.pollution_config)

        X_test = generator.add_outliers(self.config)
        y_test = self._label_outliers(self.config)

        self.data = X_train, y_train, X_test, y_test

    def _label_outliers(self, config):
        timestamps = []
        for outlier_type, outliers in config:
            for outlier in outliers:
                timestamp = outlier['timestamp'] if len(outlier['timestamp']) == 2 \
                    else (outlier['timestamp'], outlier['timestamp'] + 1)
                train_timestamps.extend(list(range(*timestamp)))

        y = np.zeros(self.length)
        return np.where(y.index.isin(timestamps), 1, 0)

    def add_missing_values(self, missing_percentage: float, per_column: bool=True):
        X_train, y_train, X_test, y_test = self.data
        if per_column:
            for col in X_train.columns:
                missing_idxs = np.random.choice(len(X_train), int(missing_percentage*len(X_train)), replace=False)
                X_train[col][missing_idxs] = np.nan
        else:
            missing_idxs = np.random.choice(len(X_train), int(missing_percentage*len(X_train)), replace=False)
            X_train.iloc[missing_idxs] = [np.nan] * X_train.shape[-1]
