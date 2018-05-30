import numpy as np
import pandas as pd
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
