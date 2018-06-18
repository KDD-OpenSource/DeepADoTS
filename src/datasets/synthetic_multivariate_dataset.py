from typing import Tuple, Callable

import numpy as np
import pandas as pd

from .dataset import Dataset


def get_noisy_value(x, strength=1):
    return x + np.random.random(np.shape(x)) * strength - strength / 2


# Use part of sinus to create a curve starting and ending with zero gradients.
# Using `length` and `amplitude` you can adjust it in both dimensions.
def get_curve(length, amplitude):
    # Transformed sinus curve: [-1, 1] -> [0, amplitude]
    def curve(t: int):
        return amplitude * (np.sin(t)/2 + 0.5)
    # Start and end of one curve section in sinus
    from_ = 1.5 * np.pi
    to_ = 3.5 * np.pi
    return np.array([curve(t) for t in np.linspace(from_, to_, length)])


# ----- Functions generating the second dimension --------- #
# A dim2 function should return a tuple containing the following three values:
# * The values of the second dimension (array of max `interval_length` numbers)
# * Starting point for the anomaly
# * End point for the anomaly section
# The last two values are ignored for generation of not anomalous data


def doubled_dim2(curve_values, anomalous, interval_length):
    factor = 4 if anomalous else 2
    return curve_values * factor, 0, len(curve_values)


def inversed_dim2(curve_values, anomalous, interval_length):
    factor = -2 if anomalous else 2
    return curve_values * factor, 0, len(curve_values)


def shrinked_dim2(curve_values, anomalous, interval_length):
    if not anomalous:
        return curve_values, -1, -1
    else:
        new_curve = curve_values[::2]
        nonce = np.zeros(len(curve_values) - len(new_curve))
        values = np.concatenate([nonce, new_curve])
        return values, 0, len(values)


def delayed_dim2(curve_values, anomalous, interval_length):
    if not anomalous:
        return curve_values, -1, -1
    else:
        # The curve in the second dimension occures a few timestamps later
        nonce = np.zeros(len(curve_values) // 10)
        values = np.concatenate([nonce, curve_values])
        return values, 0, len(values)


def xor_dim2(curve_values, anomalous, interval_length):
    orig_amplitude = max(abs(curve_values))
    orig_amplitude *= np.sign(curve_values.mean())
    pause_length = interval_length - len(curve_values)
    if not anomalous:
        # No curve during the other curve in the 1st dimension
        nonce = np.zeros(len(curve_values))
        # Insert a curve with the same amplitude during the pause of the 1st dimension
        new_curve = get_curve(pause_length, orig_amplitude)
        return np.concatenate([nonce, new_curve]), -1, -1
    else:
        # Anomaly: curves overlap (at the same time or at least half overlapping)
        max_pause = min(len(curve_values) // 2, pause_length)
        nonce = np.zeros(np.random.randint(max_pause))
        return np.concatenate([nonce, curve_values]), len(nonce), len(nonce) + len(curve_values)


class SyntheticMultivariateDataset(Dataset):

    def __init__(self, name: str = 'Synthetic Multivariate Curve Outliers',
                 length: int = 5000,
                 mean_curve_length: int = 40,  # varies between -5 and +5
                 mean_curve_amplitude: int = 1,  # By default varies between -0.5 and 1.5
                 # dim2: Lambda for curve values of 2nd dimension
                 dim2: Callable[[np.ndarray, bool, int], Tuple[np.ndarray, int, int]] = doubled_dim2,
                 pause_range: Tuple[int, int] = (5, 75),  # min and max value for this a pause
                 labels_padding: int = 6,
                 random_seed: int = 42,
                 features: int = 2,
                 file_name: str = 'synthetic_mv1.pkl'):
        super().__init__(name, file_name)
        self.length = length
        self.mean_curve_length = mean_curve_length
        self.mean_curve_amplitude = mean_curve_amplitude
        self.global_noise = 0.1  # Noise added to all dimensions over the whole timeseries
        self.dim2 = dim2
        self.pause_range = pause_range
        self.labels_padding = labels_padding
        self.random_seed = random_seed
        self.features = features

    # Randomly adjust curve size by adding noise to the passed parameters
    def get_random_curve(self, length_randomness=10, amplitude_randomness=1):
        is_negative = np.random.choice([True, False])
        sign = -1 if is_negative else 1
        new_length = get_noisy_value(self.mean_curve_length, length_randomness)
        new_amplitude = get_noisy_value(sign * self.mean_curve_amplitude, amplitude_randomness)
        return get_curve(new_length, new_amplitude)

    # The interval between two curves must be random so a detector doesn't recognize a pattern
    def create_pause(self):
        xmin, xmax = self.pause_range
        diff = xmax - xmin
        return xmin + np.random.randint(diff)

    def add_global_noise(self, x):
        return get_noisy_value(x, self.global_noise)

    """
        pollution: Portion of anomalous curves. Because it's not known how many curves there are
            in the end. It's randomly chosen based on this value. To avoid anomalies set this to zero.
    """
    def generate_data(self, pollution=0.5):
        values = np.zeros((self.length, self.features))
        labels = np.zeros(self.length)
        pos = self.create_pause()

        # First pos data points are noise (don't start directly with curve)
        values[:pos] = self.add_global_noise(values[:pos])

        while pos < self.length - self.mean_curve_length - 20:
            # General outline for the repeating curves, varying height and length
            curve = self.get_random_curve()
            # Outlier generation in second dimension
            create_anomaly = np.random.choice([False, True], p=[1-pollution, pollution])
            # After curve add pause, only noise
            end_of_interval = pos + len(curve) + self.create_pause()
            self.insert_features(values[pos:end_of_interval], labels[pos:end_of_interval], curve, create_anomaly)
            pos = end_of_interval
        # rest of values is noise
        values[pos:] = self.add_global_noise(values[pos:])
        return pd.DataFrame(values), pd.Series(labels)

    """
        Insert values for curve and following pause over all dimensions.
        interval_values is changed by reference so this function doesn't return anything.
        (this is done by using numpy place function/slice operator)

    """
    def insert_features(self, interval_values: np.ndarray, interval_labels: np.ndarray,
                        curve: np.ndarray, create_anomaly: bool):
        assert self.features == 2, 'Only two features are supported right now!'

        # Insert curve and pause in first dimension (after adding the global noise)
        interval_values[:len(curve), 0] = self.add_global_noise(curve)
        interval_values[len(curve):, 0] = self.add_global_noise(interval_values[len(curve):, 0])

        # Get values of dim2 and fill missing spots with noise
        # dim2 function gets the clean curve values (not noisy)
        interval_length = interval_values.shape[0]
        dim2_values, start, end = self.dim2(curve, create_anomaly, interval_length)
        assert len(dim2_values) <= interval_length, f'Interval too long: {len(dim2_values)} > {interval_length}'

        interval_values[:len(dim2_values), 1] = self.add_global_noise(dim2_values)
        # Fill interval up with noisy zero values
        interval_values[len(dim2_values):, 1] = self.add_global_noise(interval_values[len(dim2_values):, 1])

        # Add anomaly labels with slight padding (dont start with the first interval value).
        # The padding is curve_length / padding_factor
        if create_anomaly:
            assert end > start and start >= 0, f'Invalid anomaly indizes: {start} to {end}'
            padding = (end - start) // self.labels_padding
            interval_labels[start+padding:end-padding] += 1

    def load(self):
        np.random.seed(self.random_seed)
        X_train, y_train = self.generate_data(pollution=0)
        X_test, y_test = self.generate_data(pollution=0.5)
        self._data = X_train, y_train, X_test, y_test
