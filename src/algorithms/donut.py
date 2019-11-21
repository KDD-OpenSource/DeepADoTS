import sys

import numpy as np
import pandas as pd
import six
import tensorflow as tf
from donut import DonutTrainer, DonutPredictor, Donut as DonutModel, complete_timestamp, standardize_kpi
from donut.augmentation import MissingDataInjection
from donut.utils import BatchSlidingWindow
from tensorflow import keras as K
from tfsnippet.modules import Sequential
from tfsnippet.scaffold import TrainLoop
from tfsnippet.utils import (get_default_session_or_error,
                             ensure_variables_initialized)
from tqdm import trange

from .algorithm_utils import Algorithm, TensorflowUtils


class QuietDonutTrainer(DonutTrainer):
    def fit(self, values, labels, missing, mean, std, excludes=None,
            valid_portion=0.3, summary_dir=None):
        """
        Train the :class:`Donut` model with given data.
        From https://github.com/haowen-xu/donut/blob/master/donut/training.py but without prints.

        Args:
            values (np.ndarray): 1-D `float32` array, the standardized
                KPI observations.
            labels (np.ndarray): 1-D `int32` array, the anomaly labels.
            missing (np.ndarray): 1-D `int32` array, the indicator of
                missing points.
            mean (float): The mean of KPI observations before standardization.
            std (float): The standard deviation of KPI observations before
                standardization.
            excludes (np.ndarray): 1-D `bool` array, indicators of whether
                or not to totally exclude a point.  If a point is excluded,
                any window which contains that point is excluded.
                (default :obj:`None`, no point is totally excluded)
            valid_portion (float): Ratio of validation data out of all the
                specified training data. (default 0.3)
            summary_dir (str): Optional summary directory for
                :class:`tf.summary.FileWriter`. (default :obj:`None`,
                summary is disabled)
        """
        sess = get_default_session_or_error()

        # split the training & validation set
        values = np.asarray(values, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        missing = np.asarray(missing, dtype=np.int32)
        if len(values.shape) != 1:
            raise ValueError('`values` must be a 1-D array')
        if labels.shape != values.shape:
            raise ValueError('The shape of `labels` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(labels.shape, values.shape))
        if missing.shape != values.shape:
            raise ValueError('The shape of `missing` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(missing.shape, values.shape))

        n = int(len(values) * valid_portion)
        train_values, v_x = values[:-n], values[-n:]
        train_labels, valid_labels = labels[:-n], labels[-n:]
        train_missing, valid_missing = missing[:-n], missing[-n:]
        v_y = np.logical_or(valid_labels, valid_missing).astype(np.int32)
        if excludes is None:
            train_excludes, valid_excludes = None, None
        else:
            train_excludes, valid_excludes = excludes[:-n], excludes[-n:]

        # data augmentation object and the sliding window iterator
        # If std is zero choose a number close to zero
        aug = MissingDataInjection(mean, std, self._missing_data_injection_rate)
        train_sliding_window = BatchSlidingWindow(
            array_size=len(train_values),
            window_size=self.model.x_dims,
            batch_size=self._batch_size,
            excludes=train_excludes,
            shuffle=True,
            ignore_incomplete_batch=True,
        )
        valid_sliding_window = BatchSlidingWindow(
            array_size=len(v_x),
            window_size=self.model.x_dims,
            batch_size=self._valid_batch_size,
            excludes=valid_excludes,
        )

        # initialize the variables of the trainer, and the model
        sess.run(self._trainer_initializer)
        ensure_variables_initialized(self._train_params)

        # training loop
        lr = self._initial_lr
        # Side effect. EarlyStopping stores variables temporarely in a Temp dir
        with TrainLoop(
                param_vars=self._train_params,
                early_stopping=True,
                summary_dir=summary_dir,
                max_epoch=self._max_epoch,
                max_step=self._max_step) as loop:  # type: TrainLoop

            for epoch in loop.iter_epochs():
                x, y1, y2 = aug.augment(
                    train_values, train_labels, train_missing)
                y = np.logical_or(y1, y2).astype(np.int32)

                train_iterator = train_sliding_window.get_iterator([x, y])
                for step, (batch_x, batch_y) in loop.iter_steps(train_iterator):
                    # run a training step
                    feed_dict = dict(six.iteritems(self._feed_dict))
                    feed_dict[self._learning_rate] = lr
                    feed_dict[self._input_x] = batch_x
                    feed_dict[self._input_y] = batch_y
                    loss, _ = sess.run(
                        [self._loss, self._train_op], feed_dict=feed_dict)
                    loop.collect_metrics({'loss': loss})

                    if step % self._valid_step_freq == 0:
                        # collect variable summaries
                        if summary_dir is not None:
                            loop.add_summary(sess.run(self._summary_op))

                        # do validation in batches
                        with loop.timeit('valid_time'), loop.metric_collector('valid_loss') as mc:
                            v_it = valid_sliding_window.get_iterator([v_x, v_y])
                            for b_v_x, b_v_y in v_it:
                                feed_dict = dict(
                                    six.iteritems(self._valid_feed_dict))
                                feed_dict[self._input_x] = b_v_x
                                feed_dict[self._input_y] = b_v_y
                                loss = sess.run(self._loss, feed_dict=feed_dict)
                                mc.collect(loss, weight=len(b_v_x))

                # anneal the learning rate
                if self._lr_anneal_epochs and epoch % self._lr_anneal_epochs == 0:
                    lr *= self._lr_anneal_factor


class Donut(Algorithm, TensorflowUtils):
    """For each feature, the anomaly score is set to 1 for a point if its reconstruction probability
    is smaller than mean - std of the reconstruction probabilities for that feature. For each point
    in time, the maximum of the scores of the features is taken to support multivariate time series as well."""

    def __init__(self, num_epochs=256, batch_size=32, x_dims=120,
                 seed: int = None, gpu: int = None):
        Algorithm.__init__(self, __name__, 'Donut', seed)
        TensorflowUtils.__init__(self, seed, gpu)
        self.max_epoch = num_epochs
        self.x_dims = x_dims
        self.batch_size = batch_size
        self.means, self.stds, self.tf_sessions, self.models = [], [], [], []

    def fit(self, X: pd.DataFrame):
        with self.device:
            # Reset all results from last run to avoid reusing variables
            self.means, self.stds, self.tf_sessions, self.models = [], [], [], []
            for col_idx in trange(len(X.columns)):
                col = X.columns[col_idx]
                tf_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                timestamps = X.index
                features = X.loc[:, col].interpolate().bfill().values
                labels = pd.Series(0, X.index)
                timestamps, _, (features, labels) = complete_timestamp(timestamps, (features, labels))
                missing = np.isnan(X.loc[:, col].values)
                _, mean, std = standardize_kpi(features, excludes=np.logical_or(labels, missing))

                with tf.variable_scope('model') as model_vs:
                    model = DonutModel(
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

                trainer = QuietDonutTrainer(model=model, model_vs=model_vs, max_epoch=self.max_epoch,
                                            batch_size=self.batch_size, valid_batch_size=self.batch_size,
                                            missing_data_injection_rate=0.0, lr_anneal_factor=1.0)
                with tf_session.as_default():
                    trainer.fit(features, labels, missing, mean, std, valid_portion=0.25)
                self.means.append(mean)
                self.stds.append(std)
                self.tf_sessions.append(tf_session)
                self.models.append(model)

    def predict(self, X: pd.DataFrame):
        """Since we predict the anomaly scores for each feature independently, we already return a binarized one-
        dimensional anomaly score array."""
        with self.device:
            test_scores = np.zeros_like(X)
            for col_idx, col in enumerate(X.columns):
                mean, std, tf_session, model = \
                    self.means[col_idx], self.stds[col_idx], self.tf_sessions[col_idx], self.models[col_idx]
                test_values, _, _ = standardize_kpi(X.loc[:, col], mean=mean, std=std)
                test_missing = np.zeros_like(test_values)
                predictor = DonutPredictor(model)
                with tf_session.as_default():
                    test_score = predictor.get_score(test_values, test_missing)
                # Convert to negative reconstruction probability so score is in accordance with other detectors
                test_score = -np.power(np.e, test_score)
                test_scores[self.x_dims - 1:, col_idx] = test_score
            aggregated_test_scores = np.amax(test_scores, axis=1)
            aggregated_test_scores[:self.x_dims - 1] = np.nanmin(aggregated_test_scores) - sys.float_info.epsilon
            return aggregated_test_scores
