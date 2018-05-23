import numpy as np
import pandas as pd
import tensorflow as tf
from donut import DonutTrainer, DonutPredictor, Donut as DonutModel, complete_timestamp, standardize_kpi
from tensorflow import keras as K
from tfsnippet.modules import Sequential

from .algorithm import Algorithm


class Donut(Algorithm):
    def __init__(self):
        super(Donut).__init__()
        self.feature_col_idx, self.mean, self.std, self.model, self.model_vs, self.tf_session = \
            None, None, None, None, None, None

    def fit(self, X: pd.DataFrame, y: pd.Series, timestamp_col_name="timestamps", feature_col_idx=1):
        self.tf_session = tf.Session()
        self.feature_col_idx = feature_col_idx
        timestamps, features, labels = X[timestamp_col_name].values, X.iloc[:, self.feature_col_idx].values, y.values
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
                x_dims=120,
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
        return test_score
