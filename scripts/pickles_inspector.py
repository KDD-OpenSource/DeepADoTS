import sys
import pickle
import os
import re

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

from src.datasets import SyntheticDataGenerator
from src.evaluation import Evaluator


def load_pickles(pickles_dir):
    for filename in os.listdir(pickles_dir):
        if 'pkl' not in filename:
            print('IGNORING', os.path.join(pickles_dir, filename))
            continue
        with open(os.path.join(pickles_dir, filename), 'rb') as f:
            save_dict = pickle.load(f)
        yield save_dict


__ev__ = Evaluator([], [])


def get_optimal_threshold(*args, **kwargs):
    return __ev__.get_optimal_threshold(*args, **kwargs)


def get_dataset(ds_name, seed):
    name_regex = re.compile(r'^([^(]+)\(pol=(\d\.\d+), anom=(\d\.\d+)\)')
    match = name_regex.match(ds_name)
    pol, anom = float(match.group(2)), float(match.group(3))
    return SyntheticDataGenerator.extreme_1_polluted(seed, rel_pollution_percentage=pol, anomaly_percentage=anom)


def insert_anomaly_markers(ax, y_test, color='red', alpha=0.2):
    starts = np.arange(0, len(y_test))[(y_test == 0) & (y_test.shift(-1) == 1)]
    ends = np.arange(0, len(y_test))[(y_test == 1) & (y_test.shift(-1) == 0)]
    for start_idx, end_idx in zip(starts, ends):
        ax.axvspan(start_idx, end_idx, alpha=alpha, color=color)


def plot_score(ax, score, y_test):
    ax.plot(score)
    threshold = get_optimal_threshold(y_test, np.array(score))
    y_hat = pd.Series(Evaluator.binarize(score, threshold))
    fps = (y_hat == 1) & (y_hat != y_test)
    tps = (y_hat == 1) & (y_hat == y_test)
    insert_anomaly_markers(ax, y_test, 'gray', 0.1)
    insert_anomaly_markers(ax, tps, 'green', 0.3)
    insert_anomaly_markers(ax, fps, 'red', 0.3)
    threshold_line = len(score) * [threshold]
    ax.plot([x for x in threshold_line])


def plot_roc_curve(ax, score, y_test):
    if np.isnan(score).all():
        score = np.zeros_like(score)
    # Rank NaN below every other value in terms of anomaly score
    score[np.isnan(score)] = np.nanmin(score) - sys.float_info.epsilon
    fpr, tpr, _ = roc_curve(y_test, score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange',
            lw=2, label='area = %0.2f' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right')


def inspect_results(save_dict, ds_name, det):
    seed = save_dict['seed']
    fig, axes = plt.subplots(4, figsize=(12, 12))
    ds = get_dataset(ds_name, seed)
    X_train, y_train, X_test, y_test = ds.data()
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    axes[0].plot(X_train)
    axes[0].set_title(f'Training data for {ds_name} (seed={seed})')

    axes[1].plot(X_test)
    axes[1].set_title(f'Test data for {ds_name}')
    insert_anomaly_markers(axes[1], y_test)

    score = save_dict['results'][ds_name, det]
    plot_score(axes[2], score, y_test)
    axes[2].set_title(f'Scores for {det}')

    plot_roc_curve(axes[3], score, y_test)

    fig.tight_layout()
