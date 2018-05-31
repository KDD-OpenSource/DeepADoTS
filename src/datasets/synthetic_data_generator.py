from copy import deepcopy

from agots.generators.behavior_generators import sine_generator

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
        train_split = 0.7
        n = 1
        k = 1
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'extreme': [
                {
                    'n': 0,
                    'timestamps': [
                        (2212,), (2940,), (2510,), (2695,), (2258,), (2369,),
                        (2970,), (2319,), (2952,), (2567,), (2262,), (2512,),
                        (2589,), (2538,), (2361,), (2428,), (2343,), (2192,),
                        (2819,), (2892,)
                    ]
                }
            ]
        }
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name="Synthetic Extreme Outliers", file_name="extreme1.pkl", length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def extreme_1_polluted(pollution_percentage=0.2):
        """Full pollution -> All anomalies from test set are in train set"""
        dataset = SyntheticDataGenerator.extreme_1()

        offset = int(dataset.train_split * dataset.length)
        factor = dataset.train_split / (1 - dataset.train_split)

        pollution_config = deepcopy(dataset.outlier_config)
        anomalies = pollution_config['extreme'][0]['timestamps']
        pollution_count = min(1, pollution_percentage*len(anomalies))
        pollutions = [(int((pos - offset) * factor),) for (pos,) in anomalies[:pollution_count]]
        pollution_config['extreme'][0]['timestamps'] = pollutions

        dataset.pollution_config = pollution_config
        dataset.name = dataset.name + f'Polluted {pollution_percentage}'
        return dataset

    @staticmethod
    def extreme_1_missing(missing_percentage=0.1):
        dataset = SyntheticDataGenerator.extreme_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = dataset.name + f'Missing {missing_percentage}'
        return dataset

    @staticmethod
    def shift_1():
        length = 3000
        train_split = 0.7
        n = 1
        k = 1
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'shift': [{'n': 0, 'timestamps': [
                (2210, 2270), (2300, 2340), (2500, 2580), (2600, 2650), (2800, 2900)
            ]}],
        }
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name="Synthetic Shift Outliers", file_name="shift1.pkl", length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def shift_1_missing(missing_percentage=0.1):
        dataset = SyntheticDataGenerator.shift_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = dataset.name + f'Missing {missing_percentage}'
        return dataset

    @staticmethod
    def shift_1_polluted(pollution_percentage=0.2):
        dataset = SyntheticDataGenerator.shift_1()

        offset = int(dataset.train_split * dataset.length)
        factor = dataset.train_split / (1 - dataset.train_split)

        pollution_config = deepcopy(dataset.outlier_config)
        anomalies = pollution_config['shift'][0]['timestamps']
        pollution_count = min(1, pollution_percentage*len(anomalies))
        pollutions = [((pos1 - offset) * factor, (pos2 - offset) * factor)
                      for (pos1, pos2) in anomalies[:pollution_count]]
        pollution_config['extreme'][0]['timestamps'] = pollutions

        dataset.pollution_config = pollution_config
        dataset.name = dataset.name + f'Polluted {pollution_percentage}'
        return dataset

    @staticmethod
    def variance_1():
        length = 3000
        train_split = 0.7
        n = 1
        k = 1
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'variance': [{'n': 0, 'timestamps': [(2300, 2310), (2400, 2420), (2500, 2550), (2800, 2900)]}],
        }
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name="Synthetic Shift Outliers", file_name="variance1.pkl", length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def variance_1_missing(missing_percentage=0.1):
        dataset = SyntheticDataGenerator.variance_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = dataset.name + f'Missing {missing_percentage}'
        return dataset

    @staticmethod
    def variance_1_polluted(pollution_percentage=0.2):
        dataset = SyntheticDataGenerator.variance_1()

        offset = int(dataset.train_split * dataset.length)
        factor = dataset.train_split / (1 - dataset.train_split)

        pollution_config = deepcopy(dataset.outlier_config)
        anomalies = pollution_config['variance'][0]['timestamps']
        pollution_count = min(1, pollution_percentage*len(anomalies))
        pollutions = [((pos1 - offset) * factor, (pos2 - offset) * factor)
                      for (pos1, pos2) in anomalies[:pollution_count]]
        pollution_config['extreme'][0]['timestamps'] = pollutions

        dataset.pollution_config = pollution_config
        dataset.name = dataset.name + f'Polluted {pollution_percentage}'
        return dataset

    @staticmethod
    def trend_1():
        length = 3000
        train_split = 0.7
        n = 1
        k = 1
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'trend': [{'n': 0, 'timestamps': [(2200, 2400), (2550, 2420), (2500, 2550), (2800, 2950)]}],
        }
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name="Synthetic Shift Outliers", file_name="trend1.pkl", length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def trend_1_missing(missing_percentage=0.1):
        dataset = SyntheticDataGenerator.trend_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = dataset.name + f'Missing {missing_percentage}'
        return dataset

    @staticmethod
    def trend_1_polluted(pollution_percentage=0.2):
        dataset = SyntheticDataGenerator.trend_1()

        offset = int(dataset.train_split * dataset.length)
        factor = dataset.train_split / (1 - dataset.train_split)

        pollution_config = deepcopy(dataset.outlier_config)
        anomalies = pollution_config['trend'][0]['timestamps']
        pollution_count = min(1, pollution_percentage*len(anomalies))
        pollutions = [((pos1 - offset) * factor, (pos2 - offset) * factor)
                      for (pos1, pos2) in anomalies[:pollution_count]]
        pollution_config['extreme'][0]['timestamps'] = pollutions

        dataset.pollution_config = pollution_config
        dataset.name = dataset.name + f'Polluted {pollution_percentage}'
        return dataset

    @staticmethod
    def combined_1():
        # train begins at 2100
        length = 3000
        train_split = 0.7
        n = 1
        k = 1
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'extreme': [
                {
                    'n': 0,
                    'timestamps': [
                        (2212,), (2940,), (2510,), (2695,), (2258,), (2369,),
                        (2970,), (2319,), (2952,), (2567,), (2262,), (2512,),
                        (2589,), (2538,), (2361,), (2428,), (2343,), (2192,),
                        (2819,), (2892,)
                    ]
                }
            ],
            'shift': [{'n': 0, 'timestamps': [
                (2210, 2270), (2300, 2340), (2500, 2580), (2600, 2650), (2800, 2900)
            ]}],
            'variance': [{'n': 0, 'timestamps': [(2300, 2310), (2400, 2420), (2500, 2550), (2800, 2900)]}],
            'trend': [{'n': 0, 'timestamps': [(2200, 2400), (2550, 2420), (2500, 2550), (2800, 2950)]}]
        }
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name="Synthetic Combined Outliers", file_name="combined1.pkl", length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def combined_1_missing(missing_percentage=0.1):
        dataset = SyntheticDataGenerator.combined_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = dataset.name + f'Missing {missing_percentage}'
        return dataset

    @staticmethod
    def combined_4():
        # train begins at 2100
        length = 3000
        train_split = 0.7
        n = 4
        k = 1
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'extreme': [
                {
                    'n': 0,
                    'timestamps': [
                        (2212,), (2940,), (2510,), (2695,), (2258,), (2369,),
                        (2970,), (2319,), (2952,), (2567,), (2262,), (2512,),
                        (2589,), (2538,), (2361,), (2428,), (2343,), (2192,),
                        (2819,), (2892,)
                    ]
                }
            ],
            'shift': [{'n': 1, 'timestamps': [
                (2210, 2270), (2300, 2340), (2500, 2580), (2600, 2650), (2800, 2900)
            ]}],
            'variance': [{'n': 2, 'timestamps': [(2300, 2310), (2400, 2420), (2500, 2550), (2800, 2900)]}],
            'trend': [{'n': 3, 'timestamps': [(2200, 2400), (2550, 2420), (2500, 2550), (2800, 2950)]}]
        }
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name="Synthetic Combined Outliers 4-dimensional", file_name="combined4.pkl",
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def combined_4_missing(missing_percentage=0.1):
        dataset = SyntheticDataGenerator.combined_4()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        return dataset

    @staticmethod
    def behavior_sine_1(cycle_length=200):
        """Test seasonality (long term frequency)"""
        length = 3000
        train_split = 0.7
        n = 1
        k = 1
        shift_config = {}
        behavior = sine_generator
        behavior_config = {'cycle_duration': cycle_length, 'amplitude': 0.1}
        baseline_config = {}
        outlier_config = {}
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name="Synthetic Seasonal Outliers", file_name="sine1.pkl", length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)
