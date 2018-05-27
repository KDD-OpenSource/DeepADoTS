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

from .algorithm import Algorithm


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
                        with loop.timeit('valid_time'), \
                             loop.metric_collector('valid_loss') as mc:
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

class Donut(Algorithm):
    def __init__(self):
        super(Donut).__init__()
        self.x_dims = 120
        self.means, self.stds, self.tf_sessions, self.models = [], [], [], []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for col_idx in trange(len(X.columns)):
            col = X.columns[col_idx]
            tf_session = tf.Session()
            timestamps = X.index
            features = X.loc[:, col].values
            labels = y
            timestamps, missing, (features, labels) = complete_timestamp(timestamps, (features, labels))
            train_features, mean, std = standardize_kpi(features, excludes=np.logical_or(labels, missing))

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

            trainer = QuietDonutTrainer(model=model, model_vs=model_vs)
            with tf_session.as_default():
                trainer.fit(features, labels, missing, mean, std)
            self.means.append(mean)
            self.stds.append(std)
            self.tf_sessions.append(tf_session)
            self.models.append(model)

    def predict(self, X: pd.DataFrame):
        test_scores = np.zeros_like(X)
        prediction_mask = np.zeros(X.shape[0], dtype=bool)
        prediction_mask[self.x_dims - 1:] = 1  # The first x_dims-1 values can not be predicted
        for col_idx, col in enumerate(X.columns):
            mean, std, tf_session, model = \
                self.means[col_idx], self.stds[col_idx], self.tf_sessions[col_idx], self.models[col_idx]
            test_values, _, _ = standardize_kpi(X.loc[:, col], mean=mean, std=std)
            test_missing = np.zeros_like(test_values)
            predictor = DonutPredictor(model)
            with tf_session.as_default():
                test_score = predictor.get_score(test_values, test_missing)
            tf_session.close()
            test_score = np.power(np.e, test_score)  # Convert to reconstruction probability
            test_score = Donut.binarize(test_score)  # Binarize so 1 is an anomaly
            test_scores[self.x_dims - 1:, col_idx] = test_score
        aggregated_test_scores = np.amax(test_scores, axis=1)
        return aggregated_test_scores, prediction_mask

    @staticmethod
    def binarize(y_pred):
        threshold = np.mean(y_pred) - np.std(y_pred)
        return np.where(y_pred <= threshold, 1, 0)
