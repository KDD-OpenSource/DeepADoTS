import numpy as np
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
        df = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)
        df['y'] = np.zeros(self.length)
        baseline = df.copy()

        for timeseries in range(self.n):
            num_outliers = 10
            outlier_pos = sorted(np.random.choice(range(self.length), num_outliers, replace=False))

            timestamps = []
            for outlier in outlier_pos:
                timestamps.append((outlier,))

            df = dg.add_outliers(self.config)
            df['y'] = np.where(df.index.isin(outlier_pos), 1, 0)

        y_train = baseline['y']
        y_test = df['y']
        del baseline['y']
        del df['y']

        return baseline, y_train, df, y_test
