from typing import Tuple, Callable

import numpy as np
import pandas as pd

from .dataset import Dataset

"""
TODO:
- Parameter n_dim -> Variable Dimensionen (modulo 2)
- Type=[Inverse, Factorized, Outlier in both dimension with specific delay]
"""


def add_noise(x, strength=1):
    return x + np.random.random(np.shape(x)) * strength - strength / 2


# ----- Functions generating the second dimension --------- #


def doubled_dim2(curve_values, anomalous):
    factor = 4 if anomalous else 2
    return curve_values * factor


def inversed_dim2(curve_values, anomalous):
    factor = -2 if anomalous else 2
    return curve_values * factor


def shrinked_dim2(curve_values, anomalous):
    if not anomalous:
        return curve_values * 2
    else:
        new_curve = curve_values[::2]
        nonce = np.zeros(len(curve_values) - len(new_curve))
        return np.concatenate([nonce, new_curve])


def delayed_dim2(curve_values, anomalous):
    if not anomalous:
        return curve_values * 2
    else:
        # The curve in the second dimension occures a few timestamps later
        nonce = np.zeros(len(curve_values) // 10)
        # OLD TODO: generate method should support different arrays of various length
        return np.concatenate([nonce, curve_values])

# TODO: Add more complex outliers like delayed and XOR by defining a child class
# overwriting the generate function


class SyntheticMultivariateDataset(Dataset):

    def __init__(self, name: str = 'Synthetic Multivariate Curve Outliers',
                 length: int = 5000,
                 mean_curve_length: int = 40,  # varies between -5 and +5
                 mean_curve_amplitude: int = 1,  # By default varies between -0.5 and 1.5
                 # dim2: Lambda for curve values of 2nd dimension
                 dim2: Callable[[np.ndarray, bool], np.ndarray] = shrinked_dim2,
                 pause_length: Tuple[int, int] = (5, 75),  # min and max value for this a pause
                 random_seed: int = 42,
                 file_name: str = 'synthetic_mv1.pkl'):
        super().__init__(name, file_name)
        self.length = length
        self.mean_curve_length = mean_curve_length
        self.mean_curve_amplitude = mean_curve_amplitude
        self.global_noise = 0.1  # Noise added to all dimensions over the whole timeseries
        self.dim2 = dim2
        self.pause_length = pause_length
        self.random_seed = random_seed

    # Use part of sinus to create a curve starting and ending with zero gradients.
    # Using `length` and `amplitude` you can adjust it in both dimensions.
    def get_curve(self, length, amplitude):
        # Transformed sinus curve: [-1, 1] -> [0, amplitude]
        def curve(t: int):
            return amplitude * (np.sin(t)/2 + 0.5)
        # Start and end of one curve section in sinus
        from_ = 1.5 * np.pi
        to_ = 3.5 * np.pi
        return np.array([curve(t) for t in np.linspace(from_, to_, length)])

    # Randomly adjust curve size by adding noise to the passed parameters
    def get_random_curve(self, negative=False, length_randomness=10, amplitude_randomness=1):
        neg = -1 if negative else 1
        new_length = int(add_noise(self.mean_curve_length, length_randomness))
        new_amplitude = add_noise(neg * self.mean_curve_amplitude, amplitude_randomness)
        return self.get_curve(new_length, new_amplitude)

    # The interval between two curves must be random so a detector doesn't recognize a pattern
    def create_pause(self):
        xmin, xmax = self.pause_length
        diff = xmax - xmin
        return xmin + np.random.randint(diff)

    """
        padding_factor: Add a padding to the anomaly labels so they don't start with
            the first curve value. The padding is curve_length / padding_factor
        pollution: Portion of anomalous curves. Because it's not known how many curves there are
            in the end. It's randomly chosen based on this value. To avoid anomalies set this to zero.
    """
    def generate_data(self, padding_factor=6, pollution=0.5):
        features = 2

        values = np.zeros((self.length, features))
        labels = np.zeros(self.length)
        pos = self.create_pause()

        # First pos data points are noise (don't start directly with curve)
        values[:pos] = add_noise(values[:pos], self.global_noise)

        while pos < self.length - self.mean_curve_length - 20:
            # General outline for the repeating curves, only changes heights and length
            is_negative = np.random.choice([True, False])
            curve = self.get_random_curve(is_negative)
            # Outlier generation in second dimension
            create_anomaly = np.random.choice([False, True], p=[1-pollution, pollution])
            # Before storing in timeline add a noise
            values[pos:pos+len(curve), 0] = add_noise(curve, self.global_noise)
            # dim2 function gets the clean curve values (not noisy)
            values[pos:pos+len(curve), 1] = add_noise(
                self.dim2(curve, create_anomaly), self.global_noise)
            # Add anomaly labels with slight padding (dont start with the first value)
            if create_anomaly:
                padding = len(curve) // padding_factor
                labels[pos+padding:pos+len(curve)-padding] += 1
            pos += len(curve)
            # Add pause without curve, only noise
            pause_length = self.create_pause()
            values[pos:pos+pause_length] = add_noise(
                values[pos:pos+pause_length], self.global_noise)
            pos += pause_length
        # rest of values is noise
        values[pos:] = add_noise(values[pos:], self.global_noise)
        return pd.DataFrame(values), pd.Series(labels)

    def load(self):
        np.random.seed(self.random_seed)
        X_train, y_train = self.generate_data(pollution=0)
        X_test, y_test = self.generate_data(pollution=0.5)
        self._data = X_train, y_train, X_test, y_test
