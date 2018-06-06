import numpy as np
import pandas as pd
from agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator

from . import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, length: int=1000, n: int=4, k: int=2,
                 shift_config: dict=None,
                 behavior: object=None,
                 behavior_config: dict=None,
                 baseline_config: dict=None,
                 outlier_config: dict=None,
                 pollution_config: dict=None,
                 label_config: dict=None,
                 train_split: float=0.7,
                 random_state: int=None, **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.n = n
        self.k = k
        self.shift_config = shift_config if shift_config is not None else {}
        self.behavior = behavior
        self.behavior_config = behavior_config if behavior_config is not None else {}
        self.baseline_config = baseline_config if baseline_config is not None else {}
        self.outlier_config = outlier_config if outlier_config is not None else {}
        self.pollution_config = pollution_config if pollution_config is not None else {}
        self.label_config = label_config
        self.train_split = train_split
        self.train_test_factor = (1 - train_split) / train_split
        np.random.seed(random_state)

    def load(self):
        generator = MultivariateDataGenerator(self.length, self.n, self.k, shift_config=self.shift_config,
                                              behavior=self.behavior, behavior_config=self.behavior_config)
        generator.generate_baseline(**self.baseline_config)

        train_split_point = int(self.train_split * self.length)

        X_train = generator.add_outliers(self.pollution_config)[:train_split_point]
        y_train = self._label_outliers(self.label_config or self.pollution_config)[:train_split_point]

        X_test = generator.add_outliers(self.outlier_config)[train_split_point:]
        y_test = self._label_outliers(self.label_config or self.outlier_config)[train_split_point:]

        self._data = X_train, y_train, X_test, y_test

    def _label_outliers(self, config: dict) -> pd.Series:
        timestamps = []
        for outlier_type, outliers in config.items():
            for outlier in outliers:
                for ts in outlier['timestamps']:
                    if len(ts) == 1:  # tuple length 1
                        timestamps.append(int(*ts))
                    else:
                        timestamps.extend(list(ts))

        y = np.zeros(self.length)
        y[timestamps] = 1
        return pd.Series(y)

    def add_missing_values(self, missing_percentage: float, per_column: bool=True):
        X_train, y_train, X_test, y_test = self._data
        if per_column:
            for col in X_train.columns:
                missing_idxs = np.random.choice(len(X_train), int(missing_percentage*len(X_train)), replace=False)
                X_train[col][missing_idxs] = np.nan
        else:
            missing_idxs = np.random.choice(len(X_train), int(missing_percentage*len(X_train)), replace=False)
            X_train.iloc[missing_idxs] = [np.nan] * X_train.shape[-1]
