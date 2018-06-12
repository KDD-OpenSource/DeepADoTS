import numpy as np

from .dataset import Dataset


class SyntheticMultivariateDataset(Dataset):

    def __init__(self, name="Synthetic Multivariate Outliers"):
        super().__init__(name, file_name='synthetic_mv1.pkl')

    def noised(self, x, strength=1):
        return x + np.random.random(np.shape(x)) * strength - strength / 2

    def get_curve(self, length, amplitude):
        curve = lambda t: amplitude * (np.sin(t)/2 + 0.5)
        from_ = 1.5 * np.pi
        to_ = 3.5 * np.pi
        return np.array([curve(t) for t in np.linspace(from_, to_, length)])

    def get_random_curve(self, length, amplitude, length_randomness=10, amplitude_randomness=1):
        new_length = int(self.noised(length, length_randomness))
        new_amplitude = self.noised(amplitude, amplitude_randomness)
        return self.get_curve(new_length, new_amplitude)

    def generate_clean_data(self, T=1000, anomalous=False, pollution=0.5):
        # T: amount of timestamps
        pause = lambda: 5 + np.random.randint(70)  # Function for returning a pause interval
        mean_curve_length = 40
        mean_curve_amplitude = 1
        dim2 = lambda x: x * 2
        anomalous_dim2 = lambda x: x * self.noised(4)
        noise = 0.1

        values = np.zeros((T, 2))
        labels = np.zeros(T)
        pos = 10

        values[:pos] = self.noised(values[:pos], noise)
        while pos < T - mean_curve_length - 20:
            curve = self.get_random_curve(mean_curve_length, mean_curve_amplitude)
            create_anomaly = anomalous and np.random.choice([0, 1], p=[1-pollution, pollution])
            values[pos:pos+len(curve), 0] = self.noised(curve, noise)
            values[pos:pos+len(curve), 1] = self.noised(anomalous_dim2(curve) if create_anomaly else dim2(curve), noise)
            if create_anomaly:
                margin = len(curve) // 6
                labels[pos+margin:pos+len(curve)-margin] += 1
            pos += len(curve)
            pause_length = pause()
            values[pos:pos+pause_length] = self.noised(values[pos:pos+pause_length], noise)
            pos += pause_length
        values[pos:] = self.noised(values[pos:], noise)
        return values, labels

    def load(self):
        np.random.seed(42)
        X_train, y_train = self.generate_clean_data(1000)
        X_test, y_test = self.generate_clean_data(1000, anomalous=True)
        self._data = X_train, y_train, X_test, y_test
