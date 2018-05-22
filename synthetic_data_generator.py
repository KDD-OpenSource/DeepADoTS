import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from agots.multivariate_generators.multivariate_shift_outlier_generator import MultivariateShiftOutlierGenerator
from agots.multivariate_generators.multivariate_trend_outlier_generator import MultivariateTrendOutlierGenerator
from agots.multivariate_generators.multivariate_variance_outlier_generator import MultivariateVarianceOutlierGenerator
from agots.multivariate_generators.multivariate_data_generator import MultivariateDataGenerator
from agots.multivariate_generators.multivariate_extreme_outlier_generator import MultivariateExtremeOutlierGenerator

def generate_extreme_outliers():
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

        df = dg.add_outliers({'extreme': [{'n': timeseries, 'timestamps': timestamps}]})
        df['y'] = np.where(df.index.isin(outlier_pos), 1, 0)
    for col in df.columns:
        plt.plot(df[col], label=col)
    plt.title("X_test (synthetic extreme outliers)")
    plt.legend()
    plt.show()
    return baseline, df

def generate_shift_outliers():
    STREAM_LENGTH = 1000
    N = 4
    K = 2
    
    dg = MultivariateDataGenerator(STREAM_LENGTH, N, K, shift_config={1: 25, 2:20})
    df = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)

    for col in df.columns:
        plt.plot(df[col], label=col)
    plt.legend()
    plt.show()
    
    df['y'] = np.zeros(STREAM_LENGTH)

    for timeseries in range(N):
        num_outliers = 10
        outlier_pos = sorted(random.sample(range(STREAM_LENGTH), num_outliers))

        timestamps = []
        for i in range(0, len(outlier_pos), 2):
            timestamps.append((outlier_pos[i], outlier_pos[i+1]))

        df = dg.add_outliers({'shift': [{'n': timeseries, 'timestamps': timestamps}]})
        df[df.index.isin(outlier_pos)] = 1
    for col in df.columns:
        plt.plot(df[col], label=col)
    plt.legend()
    plt.show()
    return baseline, df


def generate_trend_outliers():
    STREAM_LENGTH = 1000
    N = 4
    K = 2
    
    dg = MultivariateDataGenerator(STREAM_LENGTH, N, K, shift_config={1: 25, 2:20})
    df = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)

    for col in df.columns:
        plt.plot(df[col], label=col)
    plt.legend()
    plt.show()
    
    df['y'] = np.zeros(STREAM_LENGTH)

    for timeseries in range(N):
        num_outliers = 10
        outlier_pos = sorted(random.sample(range(STREAM_LENGTH), num_outliers))

        timestamps = []
        for i in range(0, len(outlier_pos), 2):
            timestamps.append((outlier_pos[i], outlier_pos[i+1]))

        df = dg.add_outliers({'trend': [{'n': timeseries, 'timestamps': timestamps}]})
        df[df.index.isin(outlier_pos)] = 1
    for col in df.columns:
        plt.plot(df[col], label=col)
    plt.legend()
    plt.show()
    return baseline, df

def generate_variance_outliers():
    STREAM_LENGTH = 1000
    N = 4
    K = 2
    
    dg = MultivariateDataGenerator(STREAM_LENGTH, N, K, shift_config={1: 25, 2:20})
    df = dg.generate_baseline(initial_value_min=-4, initial_value_max=4)

    for col in df.columns:
        plt.plot(df[col], label=col)
    plt.legend()
    plt.show()
    
    df['y'] = np.zeros(STREAM_LENGTH)

    for timeseries in range(N):
        num_outliers = 10
        outlier_pos = sorted(random.sample(range(STREAM_LENGTH), num_outliers))

        timestamps = []
        for i in range(0, len(outlier_pos), 2):
            timestamps.append((outlier_pos[i], outlier_pos[i+1]))

        df = dg.add_outliers({'variance': [{'n': timeseries, 'timestamps': timestamps}]})

    for col in df.columns:
        plt.plot(df[col], label=col)
    plt.legend()
    plt.show()
    return baseline, df

def generate_outliers():
    return (generate_extreme_outliers(), generate_shift_outliers(), generate_trend_outliers(), generate_variance_outliers())