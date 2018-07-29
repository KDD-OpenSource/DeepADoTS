import pickle
import os
import re

import numpy as np

"""
    If names of algorithms changed you can rename them with this script
"""


def add_relative_pollution_param(pickles_dir):
    """
    Examples:
    '(pol=0.025, anom=0.05)',
    '(pol=0.05, anom=0.1)',
    '(pol=0.1, anom=0.2)',
    '(pol=0.2, anom=0.4)',
    '(pol=0.4, anom=0.8)',
    '(pol=0.037500000000000006, anom=0.05)',
    '(pol=0.07500000000000001, anom=0.1)',
    '(pol=0.15000000000000002, anom=0.2)',
    '(pol=0.30000000000000004, anom=0.4)',
    '(pol=0.6000000000000001, anom=0.8)',
    """
    for path in os.listdir(pickles_dir):
        print(f"Importing evaluator from '{path}'")
        with open(os.path.join(pickles_dir, path), 'rb') as f:
            save_dict = pickle.load(f)

        # save_dict['datasets'] = [remove_duplicate_bracket(x) for x in save_dict['datasets']]
        # save_dict['datasets'] = [translate_pollution_percentage(x) for x in save_dict['datasets']]
        # save_dict['benchmark_results']['dataset'] = [
        #     translate_pollution_percentage(x) for x in save_dict['benchmark_results']['dataset']]

        with open(os.path.join(pickles_dir, path), 'wb') as f:
            pickle.dump(save_dict, f)


def remove_duplicate_bracket(dataset_name):
    if dataset_name[-2:] == '))':
        return dataset_name[:-1]
    return dataset_name


def translate_pollution_percentage(dataset_name, steps=5):
    possible_values = np.linspace(0, 1, steps)  # see experiments.py:67
    name_regex = re.compile('([^\(]+)\(pol=(\d\.\d+), anom=(\d\.\d+)\)')
    match = name_regex.match(dataset_name)
    assert match is not None, 'Dataset needs to be polluted'
    absolute_pollution = float(match.group(2))
    anomaly_percentage = float(match.group(3))
    relative_pollution = absolute_pollution / anomaly_percentage if absolute_pollution > 0 else 0.0
    # To avoid rounding errors select from the only possible values by lowest distance
    relative_pollution = possible_values[np.argmin(abs(possible_values - relative_pollution))]
    return f'{match.group(1)}(pol={relative_pollution}, anom={anomaly_percentage})'


if __name__ == '__main__':
    path = os.path.join('reports', 'experiment_pollution', 'variance_1', 'evaluators')
    add_relative_pollution_param(path)
