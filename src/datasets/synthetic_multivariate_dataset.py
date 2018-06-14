import numpy as np
import pandas as pd

from .dataset import Dataset


class SyntheticMultivariateDataset(Dataset):

    def __init__(self, name="Synthetic Multivariate Curve Outliers"):
        super().__init__(name, file_name='synthetic_mv1.pkl')

    def add_noise(self, x, strength=1):
        return x + np.random.random(np.shape(x)) * strength - strength / 2

    def get_curve(self, length, amplitude):
        curve = lambda t: amplitude * (np.sin(t)/2 + 0.5)
        from_ = 1.5 * np.pi
        to_ = 3.5 * np.pi
        return np.array([curve(t) for t in np.linspace(from_, to_, length)])

    def get_random_curve(self, length, amplitude, length_randomness=10, amplitude_randomness=1):
        new_length = int(self.add_noise(length, length_randomness))
        new_amplitude = self.add_noise(amplitude, amplitude_randomness)
        return self.get_curve(new_length, new_amplitude)

    def generate_clean_data(self, T=1000, margin_factor = 6, anomalous=False, pollution=0.5):
        # T: amount of timestamps
        pause = lambda: 5 + np.random.randint(70)  # Function for returning a pause interval
        mean_curve_length = 40
        mean_curve_amplitude = 1
        dim2 = lambda x: x * 2
        anomalous_dim2 = lambda x: x * self.add_noise(4)
        noise = 0.1

        values = np.zeros((T, 2))
        labels = np.zeros(T)
        pos = pause()

        # first pos data points are noise
        values[:pos] = self.add_noise(values[:pos], noise)

        while pos < T - mean_curve_length - 20:
            # general outline for the repeating curves, only changes heights
            curve = self.get_random_curve(mean_curve_length, mean_curve_amplitude)
            # outlier generation
            create_anomaly = anomalous and np.random.choice([0, 1], p=[1-pollution, pollution])
            values[pos:pos+len(curve), 0] = self.add_noise(curve, noise)
            values[pos:pos+len(curve), 1] = self.add_noise(anomalous_dim2(curve) if create_anomaly else dim2(curve), noise)
            # add labels with slight margin
            if create_anomaly:
                margin = len(curve) // margin_factor
                labels[pos+margin:pos+len(curve)-margin] += 1
            pos += len(curve)
            pause_length = pause()
            values[pos:pos+pause_length] = self.add_noise(values[pos:pos+pause_length], noise)
            pos += pause_length
        # rest of values is noise
        values[pos:] = self.add_noise(values[pos:], noise)
        return pd.DataFrame(values), pd.Series(labels)

    def load(self):
        # T: amount of timestamps
        T = 5000
        np.random.seed(42)
        X_train, y_train = self.generate_clean_data(T)
        X_test, y_test = self.generate_clean_data(T, anomalous=True)
        self._data = X_train, y_train, X_test, y_test
