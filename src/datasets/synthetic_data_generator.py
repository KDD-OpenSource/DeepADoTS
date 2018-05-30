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
        x_test = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)
        x_test['y'] = np.zeros(self.length)
        x_train = x_test.copy()

        for timeseries in range(self.n):
            num_outliers = 10
            outlier_pos = sorted(np.random.choice(range(self.length), num_outliers, replace=False))

            timestamps = []
            for outlier in outlier_pos:
                timestamps.append((outlier,))

            x_test = dg.add_outliers(self.config)
            x_test['y'] = np.where(x_test.index.isin(outlier_pos), 1, 0)

        y_train = x_train['y']
        y_test = x_train['y']
        del x_train['y']
        del x_test['y']

        return x_train, y_train, x_test, y_test


class MissingValuesDataset(SyntheticData):

    def __init__(self, missing_percentage: float, separate: bool=False, **kwargs):
        super().__init__(**kwargs)
        self.missing_percentage = missing_percentage
        self.separate = separate

    def data(self):
        x_train, y_train, x_test, y_test = super().data()

        if self.separate:
            for col in x_train.columns:
                missing_idxs = np.random.choice(self.length, int(self.missing_percentage*self.length), replace=False)
                x_train[col][missing_idxs] = np.nan
        else:
            missing_idxs = np.random.choice(self.length, int(self.missing_percentage*self.length), replace=False)
            x_train.iloc[missing_idxs] = [np.nan] * self.n

        return x_train, y_train, x_test, y_test
