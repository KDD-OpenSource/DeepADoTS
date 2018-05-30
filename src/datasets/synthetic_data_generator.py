from .synthetic_dataset import SyntheticDataset


class SyntheticDataGenerator:
    """
    shift_config (starting at 0):
    {1: 25, 2:20}

    behavior_config for sine_generator:
    {'cycle_duration': 20
     'phase_shift': np.pi,
     'amplitude': 0.3}

    baseline_config:
    {'initial_value_min': -4,
     'initial_value_max': 4,
     'correlation_min': 0.9,
     'correlation_max': 0.7}

    outlier_config:
    {'extreme': [{'n': 0, 'timestamps': [(50,), (190,)]}],
     'shift':   [{'n': 1, 'timestamps': [(100, 190)]}],
     'trend':   [{'n': 2, 'timestamps': [(20, 150)]}],
     'variance':[{'n': 3, 'timestamps': [(50, 100)]}]}

    pollution_config:
    * see outlier_config
    """

    @staticmethod
    def extreme_1():
        # train begins at 2100
        length = 3000
        n = 1
        k = 1
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = \
            {'extreme': [{'n': 0, 'timestamps': [(2200,), (2350,), (2400,), (2780,), (2900,)]}]}
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name="no_config", file_name="empty_path", length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                random_state=random_state)

    @staticmethod
    def shift_1():
        pass

    @staticmethod
    def variance_1():
        pass

    @staticmethod
    def trend_1():
        pass

    @staticmethod
    def combined_1():
        pass

    @staticmethod
    def extreme_polluted_1():
        pass

    @staticmethod
    def variance_polluted_1():
        pass

    @staticmethod
    def trend_polluted_1():
        pass

    @staticmethod
    def combined_polluted_1():
        pass

    @staticmethod
    def behavior_extreme_1():
        pass

    @staticmethod
    def missing_1():
        pass

    @staticmethod
    def get_schwifty():
        pass
