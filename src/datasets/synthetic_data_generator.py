import numpy as np
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

    # Get a dataset by passing the method name as string. All following parameters
    # are passed through. Throws AttributeError if attribute was not found.
    @staticmethod
    def get(method, *args, **kwargs):
        func = getattr(SyntheticDataGenerator, method)
        return func(*args, **kwargs)

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
                        (2192,), (2212,), (2258,), (2262,), (2319,), (2343,),
                        (2361,), (2369,), (2428,), (2510,), (2512,), (2538,),
                        (2567,), (2589,), (2695,), (2819,), (2892,), (2940,),
                        (2952,), (2970,)
                    ]
                }
            ]
        }
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name='Synthetic Extreme Outliers', file_name='extreme1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def extreme_1_polluted(pollution_percentage=0.2):
        """Full pollution -> All anomalies from test set are in train set"""
        dataset = SyntheticDataGenerator.extreme_1()

        train_length = int(dataset.train_split * dataset.length)
        np.random.seed(123)
        indices = np.random.choice(train_length, int(pollution_percentage * train_length), replace=False)
        np.random.seed(None)
        pollution_config = {
            'extreme': [
                {
                    'n': 0,
                    'timestamps': [(i,) for i in indices]
                }
            ]
        }
        dataset.pollution_config = pollution_config

        dataset.name = f'Syn Extreme Outliers (pol={pollution_percentage})'
        return dataset

    @staticmethod
    def extreme_1_extremeness(extreme_value=10):
        """Full pollution -> All anomalies from test set are in train set"""
        dataset = SyntheticDataGenerator.extreme_1()

        dataset.outlier_config['extreme'][0]['value'] = extreme_value

        dataset.name = f'Syn Extreme Outliers (extremeness={extreme_value})'
        return dataset

    @staticmethod
    def extreme_1_missing(missing_percentage=0.1, use_zero=False):
        dataset = SyntheticDataGenerator.extreme_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage, use_zero=use_zero)
        dataset.name = f'Syn Extreme Outliers (mis={missing_percentage})'
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

        return SyntheticDataset(name='Synthetic Shift Outliers', file_name='shift1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def shift_1_missing(missing_percentage=0.1, use_zero=False):
        dataset = SyntheticDataGenerator.shift_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage, use_zero=use_zero)
        dataset.name = f'Syn Shift Outliers (mis={missing_percentage})'
        return dataset

    @staticmethod
    def shift_1_polluted(pollution_percentage=0.2):
        dataset = SyntheticDataGenerator.shift_1()

        train_length = int(dataset.train_split * dataset.length)
        np.random.seed(123)
        indices = sorted(np.random.choice(train_length, int(pollution_percentage * train_length), replace=False))
        np.random.seed(None)
        pollution_config = {
            'shift': [
                {
                    'n': 0,
                    'timestamps': [(i, j) for i, j in zip(indices[::2], indices[1::2])]
                }
            ]
        }
        dataset.pollution_config = pollution_config

        dataset.name = f'Syn Shift Outliers (pol={pollution_percentage})'
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

        return SyntheticDataset(name='Synthetic Variance Outliers', file_name='variance1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def variance_1_missing(missing_percentage=0.1, use_zero=True):
        dataset = SyntheticDataGenerator.variance_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage, use_zero=use_zero)
        dataset.name = f'Syn Variance Outliers (mis={missing_percentage})'
        return dataset

    @staticmethod
    def variance_1_polluted(pollution_percentage=0.2):
        dataset = SyntheticDataGenerator.variance_1()

        train_length = int(dataset.train_split * dataset.length)
        np.random.seed(123)
        indices = sorted(np.random.choice(train_length, int(pollution_percentage * train_length), replace=False))
        np.random.seed(None)
        pollution_config = {
            'variance': [
                {
                    'n': 0,
                    'timestamps': [(i, j) for i, j in zip(indices[::2], indices[1::2])]
                }
            ]
        }
        dataset.pollution_config = pollution_config

        dataset.name = f'Syn Variance Outliers (pol={pollution_percentage})'
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
            'trend': [{'n': 0, 'timestamps': [(2200, 2400), (2450, 2480), (2500, 2550), (2700, 2950)]}],
        }
        pollution_config = {}
        random_state = 42

        return SyntheticDataset(name='Synthetic Trend Outliers', file_name='trend1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def trend_1_missing(missing_percentage=0.1, use_zero=True):
        dataset = SyntheticDataGenerator.trend_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage, use_zero=use_zero)
        dataset.name = f'Syn Trend Outliers (mis={missing_percentage})'
        return dataset

    @staticmethod
    def trend_1_polluted(pollution_percentage=0.2):
        dataset = SyntheticDataGenerator.trend_1()

        train_length = int(dataset.train_split * dataset.length)
        np.random.seed(123)
        indices = sorted(np.random.choice(train_length, int(pollution_percentage * train_length), replace=False))
        np.random.seed(None)
        pollution_config = {
            'trend': [
                {
                    'n': 0,
                    'timestamps': [(i, j) for i, j in zip(indices[::2], indices[1::2])]
                }
            ]
        }
        dataset.pollution_config = pollution_config

        dataset.name = f'Syn Trend Outliers (pol={pollution_percentage})'
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
                        (2192,), (2212,), (2258,), (2262,), (2319,), (2343,),
                        (2361,), (2369,), (2428,), (2510,), (2512,), (2538,),
                        (2567,), (2589,), (2695,), (2819,), (2892,), (2940,),
                        (2952,), (2970,)
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

        return SyntheticDataset(name='Synthetic Combined Outliers', file_name='combined1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def combined_1_missing(missing_percentage=0.1):
        dataset = SyntheticDataGenerator.combined_1()
        dataset.load()
        dataset.add_missing_values(missing_percentage=missing_percentage)
        dataset.name = f'Syn Combined Outliers (mis={missing_percentage})'
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
                        (2192,), (2212,), (2258,), (2262,), (2319,), (2343,),
                        (2361,), (2369,), (2428,), (2510,), (2512,), (2538,),
                        (2567,), (2589,), (2695,), (2819,), (2892,), (2940,),
                        (2952,), (2970,)
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

        return SyntheticDataset(name='Synthetic Combined Outliers 4-dimensional', file_name='combined4.pkl',
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
        dataset.name = f'Syn Combined Outliers 4D (mis={missing_percentage})'
        return dataset

    @staticmethod
    def mv_extreme_1():
        # train begins at 2100
        length = 3000
        train_split = 0.7
        n = 2
        k = 2
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
        random_state = 42

        return SyntheticDataset(name="Synthetic Multivariate Extreme Outliers", file_name="mv_extreme1.pkl",
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def mv_shift_1():
        length = 3000
        train_split = 0.7
        n = 2
        k = 2
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
        random_state = 42

        return SyntheticDataset(name="Synthetic Multivariate Shift Outliers", file_name="mv_shift1.pkl",
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def mv_variance_1():
        length = 3000
        train_split = 0.7
        n = 2
        k = 2
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
        random_state = 42

        return SyntheticDataset(name="Synthetic Multivariate Variance Outliers", file_name="mv_variance1.pkl",
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def mv_trend_1():
        length = 3000
        train_split = 0.7
        n = 2
        k = 2
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
        random_state = 42

        return SyntheticDataset(name="Synthetic Multivariate Trend Outliers", file_name="mv_trend1.pkl",
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

    @staticmethod
    def mv_xor_extreme_1():
        # train begins at 2100
        length = 3000
        train_split = 0.7
        n = 2
        k = 2
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
        random_state = 42

        return SyntheticDataset(name="Synthetic Multivariate XOR Extreme Outliers", file_name="mv_xor_extreme1.pkl",
                                length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                label_config=label_config,
                                train_split=train_split, random_state=random_state)

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

        return SyntheticDataset(name='Synthetic Seasonal Outliers', file_name='sine1.pkl', length=length, n=n, k=k,
                                baseline_config=baseline_config, shift_config=shift_config,
                                behavior=behavior, behavior_config=behavior_config,
                                outlier_config=outlier_config, pollution_config=pollution_config,
                                train_split=train_split, random_state=random_state)
