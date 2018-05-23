import numpy as np
import matplotlib.pyplot as plt
import random
from agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator

class synthetic_data_generator():
    def __init__(self):
        pass
            
    def generate_outliers(self, outlier_type):
        STREAM_LENGTH = 1000
        N = 4
        K = 2

        dg = MultivariateDataGenerator(STREAM_LENGTH, N, K, shift_config={1: 25, 2:20})
        df = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)

        for col in df.columns:
            plt.plot(df[col], label=col)
        plt.title("X_train (baseline)")
        plt.legend()
        plt.show()

        df['y'] = np.zeros(STREAM_LENGTH)
        baseline = df.copy()

        for timeseries in range(N):
            num_outliers = 10
            outlier_pos = sorted(random.sample(range(STREAM_LENGTH), num_outliers))

            timestamps = []
            for outlier in outlier_pos:
                timestamps.append((outlier,))

            df = dg.add_outliers({outlier_type: [{'n': timeseries, 'timestamps': timestamps}]})
            df['y'] = np.where(df.index.isin(outlier_pos), 1, 0)
        for col in df.columns:
            plt.plot(df[col], label=col)
        plt.title("X_test (synthetic extreme outliers)")
        plt.legend()
        plt.show()
        
        y_train = baseline['y']
        y_test = df['y']
        del baseline['y']
        del df['y']

        return baseline, y_train, df, y_test