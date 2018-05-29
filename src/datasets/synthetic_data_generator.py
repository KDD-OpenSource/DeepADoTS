import numpy as np
import random
from agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator
from src.datasets.dataset import Dataset


class synthetic_data_generator(Dataset):
    def __init__(self, outlier_type="extreme"):
        self.outlier_type = outlier_type

    def get_data(self):
        STREAM_LENGTH = 1000
        N = 4
        K = 2
        np.random.seed(1337)

        dg = MultivariateDataGenerator(STREAM_LENGTH, N, K, shift_config={1: 25, 2: 20})
        df = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)

        df['y'] = np.zeros(STREAM_LENGTH)
        baseline = df.copy()

        for timeseries in range(N):
            num_outliers = 10
            outlier_pos = sorted(random.sample(range(STREAM_LENGTH), num_outliers))

            timestamps = []
            for outlier in outlier_pos:
                timestamps.append((outlier,))

            df = dg.add_outliers({self.outlier_type: [{'n': timeseries, 'timestamps': timestamps}]})
            df['y'] = np.where(df.index.isin(outlier_pos), 1, df['y'])

        y_train = baseline['y']
        y_test = df['y']
        del baseline['y']
        del df['y']

        return baseline, y_train, df, y_test
