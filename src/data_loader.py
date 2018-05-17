import numpy as np
from sklearn.model_selection import train_test_split


def get_train_test_split(data_path, data_type='npz', test_size=0.2, shuffle=False):
    data = np.load(data_path)
    if data_type == 'npz':
        data = data[data.files[0]]
    labels = data[:, -1]
    features = data[:, :-1]
    return train_test_split(features, labels, test_size=test_size, shuffle=shuffle)
