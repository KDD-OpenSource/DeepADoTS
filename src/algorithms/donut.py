import numpy as np
import pandas as pd
import tensorflow as tf
from donut import DonutTrainer, DonutPredictor, Donut as DonutModel, complete_timestamp, standardize_kpi
from tensorflow import keras as K
from tfsnippet.modules import Sequential

from .algorithm import Algorithm


class Donut(Algorithm):
    def __init__(self):
        self.name = "Donut"
        super(Donut).__init__()
        self.feature_col_idx, self.mean, self.std, self.model, self.model_vs, self.tf_session, self.x_dims = \
            None, None, None, None, None, None, 120

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_col_idx=1):
        self.feature_col_idx = feature_col_idx
        self.tf_session = tf.Session()
        timestamps = X.index
        features = X.iloc[:, self.feature_col_idx].values
        labels = y
        timestamps, missing, (features, labels) = complete_timestamp(timestamps, (features, labels))
        train_features, self.mean, self.std = standardize_kpi(features, excludes=np.logical_or(labels, missing))

        with tf.variable_scope('model') as self.model_vs:
            self.model = DonutModel(
                h_for_p_x=Sequential([
                    K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                                   activation=tf.nn.relu),
                    K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                                   activation=tf.nn.relu),
                ]),
                h_for_q_z=Sequential([
                    K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                                   activation=tf.nn.relu),
                    K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                                   activation=tf.nn.relu),
                ]),
                x_dims=self.x_dims,
                z_dims=5,
            )

        trainer = DonutTrainer(model=self.model, model_vs=self.model_vs)
        with self.tf_session.as_default():
            trainer.fit(features, labels, missing, self.mean, self.std)

    def predict(self, X):
        test_values, _, _ = standardize_kpi(X.iloc[:, self.feature_col_idx], mean=self.mean, std=self.std)
        test_missing = np.zeros_like(test_values)
        predictor = DonutPredictor(self.model)
        with self.tf_session.as_default():
            test_score = predictor.get_score(test_values, test_missing)
        self.tf_session.close()
        test_score = np.power(np.e, test_score)  # test_score is the logarithm of the probability
        test_score = np.pad(test_score, (self.x_dims - 1, 0), 'constant',
                            constant_values=np.nan)  # Prepend x_dims - 1 np.nan's
        return test_score
