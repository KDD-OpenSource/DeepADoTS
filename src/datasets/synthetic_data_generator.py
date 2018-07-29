import numpy as np
from agots.generators.behavior_generators import sine_generator

from .synthetic_dataset import SyntheticDataset

WINDOW_SIZE = 36


def generate_timestamps(start, end, percentage):
    windows = np.arange(start, end - WINDOW_SIZE, WINDOW_SIZE)
    timestamps = [(w, w+WINDOW_SIZE) for w in np.random.choice(windows, int(percentage * len(windows)), replace=False)]
    return timestamps


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

    # Get a dataset by passing the method name as string. All following parameters
    # are passed through. Throws AttributeError if attribute was not found.
    @staticmethod
    def get(method, *args, **kwargs):
        func = getattr(SyntheticDataGenerator, method)
        return func(*args, **kwargs)

    @staticmethod
    def extreme_1(seed, n=1, k=1, anomaly_percentage=0.023):
        np.random.seed(seed)
        # train begins at 2100
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}

        # outliers randomly distributed over all dimensions
        train_size = int(length * train_split)
        timestamps = [(t,) for t in np.random.randint(0, length - train_size,
                                                      int(anomaly_percentage * (length - train_size))) + train_size]

        dim = np.random.choice(n, len(timestamps))
        outlier_config = {'extreme':
                          [{'n': i, 'timestamps': [ts for d, ts in zip(dim, timestamps) if d == i]} for i in range(n)]}

        pollution_config = {}
        random_state = seed

        return SyntheticDataset(name=f'Syn Extreme Outliers (dim={n})', file_name='extreme1.pkl',
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def extreme_1_polluted(seed, rel_pollution_percentage=0.2, n=1, anomaly_percentage=0.023):
        """Full pollution -> All anomalies from test set are in train set"""
        np.random.seed(seed)
        dataset = SyntheticDataGenerator.extreme_1(seed, n)
        # Pollution as fraction of test set anomalies
        pollution_percentage = rel_pollution_percentage * anomaly_percentage

        train_size = int(dataset.length * dataset.train_split)
        timestamps = [(t,) for t in np.random.randint(0, train_size, int(pollution_percentage * train_size))]

        dim = np.random.choice(n, len(timestamps))
        pollution_config = {'extreme': [{'n': i, 'timestamps':
                                         [ts for d, ts in zip(dim, timestamps) if d == i]} for i in range(n)]}
        dataset.pollution_config = pollution_config

        dataset.name = f'Syn Extreme Outliers (pol={rel_pollution_percentage}, anom={anomaly_percentage})'
        return dataset

    @staticmethod
    def extreme_1_extremeness(seed, extreme_value=10, n=1):
        """Full pollution -> All anomalies from test set are in train set"""
        dataset = SyntheticDataGenerator.extreme_1(seed, n)

        dataset.outlier_config['extreme'][0]['factor'] = extreme_value

        dataset.name = f'Syn Extreme Outliers (extremeness={extreme_value})'
        return dataset

    @staticmethod
    def extreme_1_missing(seed, missing_percentage=0.1, n=1):
        dataset = SyntheticDataGenerator.extreme_1(seed, n)
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = f'Syn Extreme Outliers (mis={missing_percentage})'
        return dataset

    @staticmethod
    def shift_1(seed, n=1, k=1, anomaly_percentage=0.2):
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}

        timestamps = generate_timestamps(int(train_split * length), length, anomaly_percentage)

        # outliers randomly distributed over all dimensions
        dim = np.random.choice(n, len(timestamps))
        outlier_config = {'shift':
                          [{'n': i, 'timestamps': [ts for d, ts in zip(dim, timestamps) if d == i]} for i in range(n)]}

        pollution_config = {}
        random_state = seed

        return SyntheticDataset(name=f'Syn Shift Outliers (dim={n})', file_name='shift1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def shift_1_missing(seed, missing_percentage=0.1, n=1):
        dataset = SyntheticDataGenerator.shift_1(seed, n)
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = f'Syn Shift Outliers (mis={missing_percentage})'
        return dataset

    @staticmethod
    def shift_1_polluted(seed, rel_pollution_percentage=0.2, n=1, anomaly_percentage=0.2):
        np.random.seed(seed)
        dataset = SyntheticDataGenerator.shift_1(seed, n, anomaly_percentage=anomaly_percentage)
        # Pollution as fraction of test set anomalies
        pollution_percentage = rel_pollution_percentage * anomaly_percentage

        timestamps = generate_timestamps(0, int(dataset.train_split * dataset.length), pollution_percentage)

        dim = np.random.choice(n, len(timestamps))
        pollution_config = {'shift': [{'n': i, 'timestamps':
                                       [ts for d, ts in zip(dim, timestamps) if d == i]} for i in range(n)]}
        dataset.pollution_config = pollution_config

        dataset.name = f'Syn Shift Outliers (pol={rel_pollution_percentage}, anom={anomaly_percentage}))'
        return dataset

    @staticmethod
    def shift_1_extremeness(seed, extreme_value=10, n=1):
        """Full pollution -> All anomalies from test set are in train set"""
        dataset = SyntheticDataGenerator.shift_1(seed, n)

        dataset.outlier_config['shift'][0]['factor'] = extreme_value

        dataset.name = f'Syn Shift Outliers (extremeness={extreme_value})'
        return dataset

    @staticmethod
    def variance_1(seed, n=1, k=1, anomaly_percentage=0.2):
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}

        timestamps = generate_timestamps(int(train_split * length), length, anomaly_percentage)

        # outliers randomly distributed over all dimensions
        dim = np.random.choice(n, len(timestamps))
        outlier_config = {'variance':
                          [{'n': i, 'timestamps': [ts for d, ts in zip(dim, timestamps) if d == i]} for i in range(n)]}

        pollution_config = {}
        random_state = seed

        return SyntheticDataset(name=f'Syn Variance Outliers (dim={n})', file_name='variance1.pkl',
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def variance_1_missing(seed, missing_percentage=0.1, n=1):
        dataset = SyntheticDataGenerator.variance_1(seed, n)
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = f'Syn Variance Outliers (mis={missing_percentage})'
        return dataset

    @staticmethod
    def variance_1_polluted(seed, rel_pollution_percentage=0.2, n=1, anomaly_percentage=0.2):
        np.random.seed(seed)
        dataset = SyntheticDataGenerator.variance_1(seed, n)
        # Pollution as fraction of test set anomalies
        pollution_percentage = rel_pollution_percentage * anomaly_percentage

        timestamps = generate_timestamps(0, int(dataset.train_split * dataset.length), pollution_percentage)

        dim = np.random.choice(n, len(timestamps))
        pollution_config = {'variance': [{'n': i, 'timestamps':
                                          [ts for d, ts in zip(dim, timestamps) if d == i]} for i in range(n)]}
        dataset.pollution_config = pollution_config

        dataset.name = f'Syn Variance Outliers (pol={rel_pollution_percentage}, anom={anomaly_percentage}))'
        return dataset

    @staticmethod
    def variance_1_extremeness(seed, extreme_value=10, n=1):
        """Full pollution -> All anomalies from test set are in train set"""
        dataset = SyntheticDataGenerator.variance_1(seed, n)

        dataset.outlier_config['variance'][0]['factor'] = extreme_value

        dataset.name = f'Syn Variance Outliers (extremeness={extreme_value})'
        return dataset

    @staticmethod
    def trend_1(seed, n=1, k=1, anomaly_percentage=0.2):
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}

        timestamps = generate_timestamps(int(train_split * length), length, anomaly_percentage)

        dim = np.random.choice(n, len(timestamps))
        outlier_config = {'trend':
                          [{'n': i, 'timestamps': [ts for d, ts in zip(dim, timestamps) if d == i]} for i in range(n)]}

        pollution_config = {}
        random_state = seed

        return SyntheticDataset(name=f'Syn Trend Outliers (dim={n})', file_name='trend1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def trend_1_missing(seed, missing_percentage=0.1, n=1):
        dataset = SyntheticDataGenerator.trend_1(seed, n)
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = f'Syn Trend Outliers (mis={missing_percentage})'
        return dataset

    @staticmethod
    def trend_1_polluted(seed, rel_pollution_percentage=0.2, n=1, anomaly_percentage=0.2):
        np.random.seed(seed)
        dataset = SyntheticDataGenerator.trend_1(seed, n)
        # Pollution as fraction of test set anomalies
        pollution_percentage = rel_pollution_percentage * anomaly_percentage

        timestamps = generate_timestamps(0, int(dataset.train_split * dataset.length), pollution_percentage)

        dim = np.random.choice(n, len(timestamps))
        pollution_config = {'trend': [{'n': i, 'timestamps':
                                       [ts for d, ts in zip(dim, timestamps) if d == i]} for i in range(n)]}
        dataset.pollution_config = pollution_config

        dataset.name = f'Syn Trend Outliers (pol={rel_pollution_percentage}, anom={anomaly_percentage}))'
        return dataset

    @staticmethod
    def trend_1_extremeness(seed, extreme_value=10, n=1):
        """Full pollution -> All anomalies from test set are in train set"""
        dataset = SyntheticDataGenerator.trend_1(seed, n)

        dataset.outlier_config['trend'][0]['factor'] = extreme_value

        dataset.name = f'Syn Trend Outliers (extremeness={extreme_value})'
        return dataset

    @staticmethod
    def combined_1(seed, n=1, k=1):
        # train begins at 2100
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        timestamps_ext = [(2192,), (2212,), (2258,), (2262,), (2319,), (2343,),
                          (2361,), (2369,), (2428,), (2510,), (2512,), (2538,),
                          (2567,), (2589,), (2695,), (2819,), (2892,), (2940,),
                          (2952,), (2970,)]
        dim_ext = np.random.choice(n, len(timestamps_ext))
        timestamps_shi = [(2210, 2270), (2300, 2340), (2500, 2580), (2600, 2650), (2800, 2900)]
        dim_shi = np.random.choice(n, len(timestamps_shi))
        timestamps_var = [(2300, 2310), (2400, 2420), (2500, 2550), (2800, 2900)]
        dim_var = np.random.choice(n, len(timestamps_var))
        timestamps_tre = [(2200, 2400), (2400, 2420), (2500, 2550), (2800, 2950)]
        dim_tre = np.random.choice(n, len(timestamps_tre))
        outlier_config = {'extreme': [{'n': i, 'timestamps':
                                       [ts for d, ts in zip(dim_ext, timestamps_ext) if d == i]} for i in range(n)],
                          'shift': [{'n': i, 'timestamps':
                                     [ts for d, ts in zip(dim_shi, timestamps_shi) if d == i]} for i in range(n)],
                          'variance': [{'n': i, 'timestamps':
                                        [ts for d, ts in zip(dim_var, timestamps_var) if d == i]} for i in range(n)],
                          'trend': [{'n': i, 'timestamps':
                                     [ts for d, ts in zip(dim_tre, timestamps_tre) if d == i]} for i in range(n)]}
        pollution_config = {}
        random_state = seed

        return SyntheticDataset(name=f'Synthetic Combined Outliers (dim={n})', file_name='combined1.pkl', length=length,
                                n=n, k=k, baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def combined_1_missing(seed, missing_percentage=0.1, n=1):
        dataset = SyntheticDataGenerator.combined_1(seed, n)
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = f'Syn Combined Outliers (mis={missing_percentage})'
        return dataset

    @staticmethod
    def combined_4(seed, n=4, k=4):
        # train begins at 2100
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}

        indices = range(0, n+1, n//4)
        outl_chunks = [np.arange(j, indices[i+1]) for i, j in enumerate(indices[:-1])]
        timestamps_ext = [(2192,), (2212,), (2258,), (2262,), (2319,), (2343,),
                          (2361,), (2369,), (2428,), (2510,), (2512,), (2538,),
                          (2567,), (2589,), (2695,), (2819,), (2892,), (2940,),
                          (2952,), (2970,)]
        dim_ext = np.random.choice(outl_chunks[0], len(timestamps_ext))
        timestamps_shi = [(2210, 2270), (2300, 2340), (2500, 2580), (2600, 2650), (2800, 2900)]
        dim_shi = np.random.choice(outl_chunks[1], len(timestamps_shi))
        timestamps_var = [(2300, 2310), (2400, 2420), (2500, 2550), (2800, 2900)]
        dim_var = np.random.choice(outl_chunks[2], len(timestamps_var))
        timestamps_tre = [(2200, 2400), (2550, 2420), (2500, 2550), (2800, 2950)]
        dim_tre = np.random.choice(outl_chunks[3], len(timestamps_tre))
        outlier_config = {'extreme': [{'n': i, 'timestamps':
                                       [ts for d, ts in zip(dim_ext, timestamps_ext) if d == i]} for i in range(n)],
                          'shift': [{'n': i, 'timestamps':
                                     [ts for d, ts in zip(dim_shi, timestamps_shi) if d == i]} for i in range(n)],
                          'variance': [{'n': i, 'timestamps':
                                        [ts for d, ts in zip(dim_var, timestamps_var) if d == i]} for i in range(n)],
                          'trend': [{'n': i, 'timestamps':
                                     [ts for d, ts in zip(dim_tre, timestamps_tre) if d == i]} for i in range(n)]}
        pollution_config = {}
        random_state = seed

        return SyntheticDataset(name=f'Synthetic Combined Outliers (dim={n})', file_name='combined4.pkl',
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def combined_4_missing(seed, missing_percentage=0.1, n=4):
        dataset = SyntheticDataGenerator.combined_4(seed, n)
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = f'Syn Combined Outliers 4D (mis={missing_percentage})'
        return dataset

# for accurate high-dimensional multivariate datasets (>2) rather use synthetic_multivariate_dataset.py
    @staticmethod
    def mv_extreme_1(seed, n=2, k=2):
        # train begins at 2100
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'extreme': [
                {
                    'n': 0,
                    'timestamps': [
                        (2192,), (2212,), (2258,), (2262,), (2319,), (2343,),
                        (2361,), (2369,), (2428,), (2510,), (2512,), (2538,),
                        (2567,), (2589,), (2695,), (2819,), (2892,), (2940,),
                        (2952,), (2970,)
                    ]
                },
                {
                    'n': 1,
                    'timestamps': [
                        (2192,), (2212,), (2258,), (2262,), (2319,), (2343,),
                        (2361,), (2369,), (2428,), (2510,), (2512,), (2538,),
                        (2567,), (2589,), (2819,), (2892,), (2940,),
                        (2952,), (2970,)
                    ]
                }
            ]
        }
        pollution_config = {
            'extreme': [
                {
                    'n': 0,
                    'timestamps': [
                        (222,), (254,), (258,), (317,), (322,), (366,), (399,),
                        (736,), (769,), (770,), (784,), (795,), (819,), (842,),
                        (1163,), (1214,), (1319,), (1352,), (1366,), (1485,),
                        (1622,), (1639,), (1676,), (1686,), (1770,), (1820,),
                        (1877,), (1913,), (1931,), (2070,)
                    ]
                },
                {
                    'n': 1,
                    'timestamps': [
                        (222,), (254,), (258,), (317,), (322,), (366,), (399,),
                        (736,), (769,), (770,), (784,), (795,), (819,), (842,),
                        (1163,), (1214,), (1319,), (1352,), (1366,), (1485,),
                        (1622,), (1639,), (1676,), (1686,), (1770,), (1820,),
                        (1877,), (1913,), (1931,), (2070,)
                    ]
                }
            ]
        }
        label_config = {
            'outlier_at': [
                {
                    'timestamps': [
                        (2695,)
                    ]
                }
            ]
        }
        random_state = seed

        return SyntheticDataset(name=f'Synthetic Multivariate Extreme Outliers (dim={n})', file_name='mv_extreme1.pkl',
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def mv_shift_1(seed, n=2, k=2):
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'shift': [
                {'n': 0, 'timestamps': [
                    (2210, 2270), (2300, 2340), (2500, 2580), (2600, 2650), (2800, 2900)
                ]},
                {'n': 1, 'timestamps': [
                    (2210, 2270), (2500, 2580), (2800, 2900)
                ]}
            ]
        }
        pollution_config = {
            'shift': [
                {'n': 0, 'timestamps': [
                    (152, 248), (229, 285), (707, 779), (720, 754), (836, 901),
                    (847, 928), (883, 989), (922, 987), (1258, 1351), (1289, 1340),
                    (1401, 1424), (1717, 1742), (1808, 1836), (1828, 1895),
                    (1830, 1891)]},
                {'n': 1, 'timestamps': [
                    (152, 248), (229, 285), (707, 779), (720, 754), (836, 901),
                    (847, 928), (883, 989), (922, 987), (1258, 1351), (1289, 1340),
                    (1401, 1424), (1717, 1742), (1808, 1836), (1828, 1895),
                    (1830, 1891)]}
            ]
        }
        label_config = {
            'outlier_at': [
                {
                    'timestamps': [
                        (2300, 2340), (2600, 2650)
                    ]
                }
            ]
        }
        random_state = seed

        return SyntheticDataset(name=f'Synthetic Multivariate Shift Outliers (dim={n})', file_name='mv_shift1.pkl',
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def mv_variance_1(seed, n=2, k=2):
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'variance': [{'n': 0, 'timestamps': [(2300, 2310), (2400, 2420), (2500, 2550), (2800, 2900)]},
                         {'n': 1, 'timestamps': [(2300, 2310), (2400, 2420), (2800, 2900)]}],
        }
        pollution_config = {
            'variance': [
                {'n': 0, 'timestamps': [
                    (171, 280), (482, 675), (1104, 1366), (1519, 1724), (1996, 2159)]},
                {'n': 1, 'timestamps': [
                    (171, 280), (482, 675), (1104, 1366), (1519, 1724), (1996, 2159)]},
            ]
        }
        label_config = {
            'outlier_at': [
                {
                    'timestamps': [
                        (2500, 2550)
                    ]
                }
            ]
        }
        random_state = seed

        return SyntheticDataset(name=f'Synthetic Multivariate Variance Outliers (dim={n})',
                                file_name='mv_variance1.pkl',
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def mv_trend_1(seed, n=2, k=2):
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'trend': [{'n': 0, 'timestamps': [(2200, 2400), (2450, 2480), (2700, 2950)]},
                      {'n': 1, 'timestamps': [(2200, 2400), (2450, 2480), (2500, 2550), (2700, 2950)]}]
        }
        pollution_config = {
            'trend': [{'n': 0, 'timestamps': [(200, 400), (550, 420), (500, 550), (1200, 1350)]},
                      {'n': 1, 'timestamps': [(200, 400), (550, 420), (500, 550), (1200, 1350)]}]
        }
        label_config = {
            'outlier_at': [
                {
                    'timestamps': [
                        (2500, 2550)
                    ]
                }
            ]
        }
        random_state = seed

        return SyntheticDataset(name=f'Synthetic Multivariate Trend Outliers (dim={n})', file_name='mv_trend1.pkl',
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def mv_xor_extreme_1(seed, n=2, k=2):
        # train begins at 2100
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = None
        behavior_config = {}
        baseline_config = {}
        outlier_config = {
            'extreme': [
                {
                    'n': 0,
                    'timestamps': [
                        (2192,), (2262,), (2319,),
                        (2361,), (2428,), (2512,),
                        (2567,), (2695,), (2892,),
                        (2952,)
                    ]
                },
                {
                    'n': 1,
                    'timestamps': [
                        (2212,), (2262,), (2343,),
                        (2369,), (2510,), (2538,),
                        (2589,), (2819,), (2940,),
                        (2952,), (2970,)
                    ]
                }
            ]
        }
        pollution_config = {
            'extreme': [
                {
                    'n': 0,
                    'timestamps': [
                        (222,), (258,), (322,), (399,),
                        (769,), (784,), (819,),
                        (1214,), (1352,), (1485,),
                        (1639,), (1686,), (1820,),
                        (1913,), (2070,)
                    ]
                },
                {
                    'n': 1,
                    'timestamps': [
                        (254,), (317,), (366,),
                        (736,), (770,), (795,), (842,),
                        (1163,), (1319,), (1366,),
                        (1622,), (1676,), (1770,),
                        (1877,), (1931,),
                    ]
                }
            ]
        }
        label_config = {
            'outlier_at': [
                {
                    'timestamps': [
                        (2262,), (2952,),
                    ]
                }
            ]
        }
        random_state = seed

        return SyntheticDataset(name=f'Synthetic Multivariate XOR Extreme Outliers (dim={n})',
                                file_name='mv_xor_extreme1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def behavior_sine_1(seed, cycle_length=200, n=1, k=1):
        """Test seasonality (long term frequency)"""
        length = 3000
        train_split = 0.7
        shift_config = {}
        behavior = sine_generator
        behavior_config = {'cycle_duration': cycle_length, 'amplitude': 0.1}
        baseline_config = {}
        outlier_config = {}
        pollution_config = {}
        random_state = seed

        return SyntheticDataset(name='Synthetic Seasonal Outliers (dim={n})', file_name='sine1.pkl', length=length,
                                n=n, k=k, baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)
