import logging
from typing import Tuple, Callable

import numpy as np
import pandas as pd

from . import Dataset


class SyntheticMultivariateDataset(Dataset):
    def __init__(self,
                 # anomaly_func: Lambda for curve values of 2nd dimension
                 anomaly_func: Callable[[np.ndarray, bool, int], Tuple[np.ndarray, int, int]],
                 name: str = 'Synthetic Multivariate Curve Outliers',
                 length: int = 5000,
                 mean_curve_length: int = 40,  # varies between -5 and +5
                 mean_curve_amplitude: int = 1,  # By default varies between -0.5 and 1.5
                 pause_range: Tuple[int, int] = (5, 75),  # min and max value for this a pause
                 labels_padding: int = 6,
                 random_seed: int = None,
                 features: int = 2,
                 group_size: int = None,
                 test_pollution: float = 0.5,
                 train_pollution: float = 0,
                 global_noise: float = 0.1,  # Noise added to all dimensions over the whole timeseries
                 file_name: str = 'synthetic_mv1.pkl'):
        super().__init__(f'{name} (f={anomaly_func.__name__})', file_name)
        self.length = length
        self.mean_curve_length = mean_curve_length
        self.mean_curve_amplitude = mean_curve_amplitude
        self.global_noise = global_noise
        self.anomaly_func = anomaly_func
        self.pause_range = pause_range
        self.labels_padding = labels_padding
        self.random_seed = random_seed
        self.test_pollution = test_pollution
        self.train_pollution = train_pollution

        assert features >= 2, 'At least two dimensions are required for generating MV outliers'
        self.features = features
        assert group_size is None or (features >= group_size > 0), 'Group size may not be greater ' \
                                                                   'than amount of dimensions'
        self.group_size = group_size or self.features
        assert self.group_size <= self.features
        if self.features % self.group_size == 1:  # How many dimensions each correlated group has
            logging.warn('Group size results in one overhanging univariate group. Generating multivariate'
                         'anomalies on univariate data is impossible.')

        if self.train_pollution > 0:
            self.name = self.name + f'(pol={self.train_pollution})'

    @staticmethod
    def get_noisy_value(x, strength=1):
        return x + np.random.random(np.shape(x)) * strength - strength / 2

    # Use part of sinus to create a curve starting and ending with zero gradients.
    # Using `length` and `amplitude` you can adjust it in both dimensions.
    @staticmethod
    def get_curve(length, amplitude):
        # Transformed sinus curve: [-1, 1] -> [0, amplitude]
        def curve(t: int):
            return amplitude * (np.sin(t) / 2 + 0.5)

        # Start and end of one curve section in sinus
        from_ = 1.5 * np.pi
        to_ = 3.5 * np.pi
        return np.array([curve(t) for t in np.linspace(from_, to_, length)])

    # Randomly adjust curve size by adding noise to the passed parameters
    def get_random_curve(self, length_randomness=10, amplitude_randomness=1):
        is_negative = np.random.choice([True, False])
        sign = -1 if is_negative else 1
        new_length = self.get_noisy_value(self.mean_curve_length, length_randomness)
        new_amplitude = self.get_noisy_value(sign * self.mean_curve_amplitude, amplitude_randomness)
        return self.get_curve(new_length, new_amplitude)

    # The interval between two curves must be random so a detector doesn't recognize a pattern
    def create_pause(self):
        xmin, xmax = self.pause_range
        diff = xmax - xmin
        return xmin + np.random.randint(diff)

    def add_global_noise(self, x):
        return self.get_noisy_value(x, self.global_noise)

    def generate_correlated_group(self, dimensions, pollution):
        values = np.zeros((self.length, dimensions))
        labels = np.zeros(self.length)

        # First pos data points are noise (don't start directly with curve)
        pos = self.create_pause()
        values[:pos, :] = self.add_global_noise(values[:pos])

        # Keep space (20) at the end
        while pos < self.length - self.mean_curve_length - 20:
            # General outline for the repeating curves, varying height and length
            curve = self.get_random_curve()
            # Outlier generation in second dimension
            create_anomaly = np.random.choice([False, True], p=[1 - pollution, pollution])
            # After curve add pause, only noise
            end_of_interval = pos + len(curve) + self.create_pause()
            self.insert_features(values[pos:end_of_interval, :], labels[pos:end_of_interval], curve, create_anomaly)
            pos = end_of_interval
        # rest of values is noise
        values[:pos, :] = self.add_global_noise(values[:pos, :])
        return pd.DataFrame(values), pd.Series(labels)

    """
        pollution: Portion of anomalous curves. Because it's not known how many curves there are
            in the end. It's randomly chosen based on this value. To avoid anomalies set this to zero.
    """

    def generate_data(self, pollution):
        value_dfs, label_series = [], []
        for i in range(0, self.features, self.group_size):
            values, labels = self.generate_correlated_group(min(self.group_size, self.features - i),
                                                            pollution=pollution * self.group_size / self.features)
            value_dfs.append(values)
            label_series.append(labels)
        labels = pd.Series(np.logical_or.reduce(label_series))
        values = pd.concat(value_dfs, axis=1, ignore_index=True)
        return values, labels

    """
        Insert values for curve and following pause over all dimensions.
        interval_values is changed by reference so this function doesn't return anything.
        (this is done by using numpy place function/slice operator)

    """

    def insert_features(self, interval_values: np.ndarray, interval_labels: np.ndarray,
                        curve: np.ndarray, create_anomaly: bool):
        # Randomly switch between dimensions for inserting the anomaly_func
        anomaly_dim = np.random.randint(1, interval_values.shape[1])

        # Insert curve and pause in first dimension (after adding the global noise)
        for i in set(range(interval_values.shape[1])) - {anomaly_dim}:
            # Add to interval_values because they already contain a shift based on the dimension
            interval_values[:len(curve), i] += self.add_global_noise(curve)
            interval_values[len(curve):, i] = self.add_global_noise(interval_values[len(curve):, i])

        # Get values of anomaly_func and fill missing spots with noise
        # anomaly_func function gets the clean curve values (not noisy)
        interval_length = interval_values.shape[0]
        anomaly_values, start, end = self.anomaly_func(curve, create_anomaly, interval_length)
        assert len(anomaly_values) <= interval_length, f'Interval too long: {len(anomaly_values)} > {interval_length}'

        interval_values[:len(anomaly_values), anomaly_dim] += self.add_global_noise(anomaly_values)
        # Fill interval up with noisy zero values
        interval_values[len(anomaly_values):, anomaly_dim] = self.add_global_noise(
            interval_values[len(anomaly_values):, anomaly_dim])

        # Add anomaly labels with slight padding (dont start with the first interval value).
        # The padding is curve_length / padding_factor
        if create_anomaly:
            assert end > start and start >= 0, f'Invalid anomaly indizes: {start} to {end}'
            padding = (end - start) // self.labels_padding
            interval_labels[start + padding:end - padding] += 1

    def load(self):
        np.random.seed(self.random_seed)
        X_train, y_train = self.generate_data(pollution=self.train_pollution)
        X_test, y_test = self.generate_data(pollution=self.test_pollution)
        self._data = X_train, y_train, X_test, y_test
