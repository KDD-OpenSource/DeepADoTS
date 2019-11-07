import numpy as np
import tensorflow as tf
from tqdm import trange

from .algorithm_utils import Algorithm, TensorflowUtils


class RecurrentEBM(Algorithm, TensorflowUtils):
    """ Recurrent Energy-Based Model implementation using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, num_epochs=100, n_hidden=50, n_hidden_recurrent=100,
                 min_lr=1e-3, min_energy=None, batch_size=10,
                 seed: int = None, gpu: int = None):
        Algorithm.__init__(self, __name__, 'Recurrent EBM', seed)
        TensorflowUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.n_hidden = n_hidden  # Size of RBM's hidden layer
        self.n_hidden_recurrent = n_hidden_recurrent  # Size of RNN's hidden layer
        self.min_lr = min_lr
        self.min_energy = min_energy  # Threshold for anomaly
        self.batch_size = batch_size

        # Placeholders
        self.input_data = None
        self.lr = None
        self._batch_size = None

        # Variables
        self.W, self.Wuh, self.Wux, self.Wxu, self.Wuu, self.bu, self.u0, self.bh, self.bx, self.BH_t, self.BX_t = [None] * 11
        self.tvars = []

        self.update = None
        self.cost = None

        self.tf_session = None

    def fit(self, X):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        with self.device:
            self._build_model(X.shape[1])
            self.tf_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self._initialize_tf()
            self._train_model(X, self.batch_size)

    def predict(self, X):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        with self.device:
            scores = []
            labels = []

            for i in range(len(X)):
                reconstruction_err = self.tf_session.run([self.cost],
                                                         feed_dict={self.input_data: X[i:i + 1],
                                                                    self._batch_size: 1})
                scores.append(reconstruction_err[0])
            if self.min_energy is not None:
                labels = np.where(scores >= self.min_energy)

            scores = np.array(scores)

            return (labels, scores) if self.min_energy is not None else scores

    def _train_model(self, train_set, batch_size):
        for epoch in trange(self.num_epochs):
            costs = []
            for i in range(0, len(train_set), batch_size):
                x = train_set[i:i + batch_size]
                if len(x) == batch_size:
                    alpha = self.min_lr  # min(self.min_lr, 0.1 / float(i + 1))
                    _, C = self.tf_session.run([self.update, self.cost],
                                               feed_dict={self.input_data: x, self.lr: alpha,
                                                          self._batch_size: batch_size})
                    costs.append(C)
            self.logger.debug(f'Epoch: {epoch+1} Cost: {np.mean(costs)}')

    def _initialize_tf(self):
        init = tf.global_variables_initializer()
        self.tf_session.run(init)

    def _build_model(self, n_visible):
        self.input_data, self.lr, self._batch_size = self._create_placeholders(n_visible)
        self.W, self.Wuh, self.Wux, self.Wxu, self.Wuu, self.bu, \
        self.u0, self.bh, self.bx, self.BH_t, self.BX_t = self._create_variables(n_visible)

        def rnn_recurrence(u_tmin1, sl):
            # Iterate through the data in the batch and generate the values of the RNN hidden nodes
            sl = tf.reshape(sl, [1, n_visible])
            u_t = tf.nn.softplus(self.bu + tf.matmul(sl, self.Wxu) + tf.matmul(u_tmin1, self.Wuu))
            return u_t

        def visible_bias_recurrence(bx_t, u_tmin1):
            # Iterate through the values of the RNN hidden nodes and generate the values of the visible bias vectors
            bx_t = tf.add(self.bx, tf.matmul(u_tmin1, self.Wux))
            return bx_t

        def hidden_bias_recurrence(bh_t, u_tmin1):
            # Iterate through the values of the RNN hidden nodes and generate the values of the hidden bias vectors
            bh_t = tf.add(self.bh, tf.matmul(u_tmin1, self.Wuh))
            return bh_t

        self.BH_t = tf.tile(self.BH_t, [self._batch_size, 1])
        self.BX_t = tf.tile(self.BX_t, [self._batch_size, 1])

        # Scan through the rnn and generate the value for each hidden node in the batch
        u_t = tf.scan(rnn_recurrence, self.input_data, initializer=self.u0)
        # Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
        self.BX_t = tf.reshape(tf.scan(visible_bias_recurrence, u_t, tf.zeros([1, n_visible], tf.float32)),
                               [n_visible, self._batch_size])
        self.BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, u_t, tf.zeros([1, self.n_hidden], tf.float32)),
                               [self.n_hidden, self._batch_size])

        self.cost = self._run_ebm(self.input_data, self.W, self.BX_t, self.BH_t)

        self.tvars = [self.W, self.Wuh, self.Wux, self.Wxu, self.Wuu, self.bu, self.u0, self.bh, self.bx]
        opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars), 1)
        self.update = opt_func.apply_gradients(zip(grads, self.tvars))

    def _run_ebm(self, x, W, b_prime, b):
        """ Runs EBM for time step and returns reconstruction error.
        1-layer implementation, TODO: implement and test deep structure
        """
        x = tf.transpose(x)  # For batch processing

        forward = tf.matmul(tf.transpose(W), x) + b
        reconstruction = tf.matmul(W, tf.sigmoid(forward)) + b_prime
        loss = tf.reduce_sum(tf.square(x - reconstruction))
        return loss

    def _create_placeholders(self, n_visible):
        x = tf.placeholder(tf.float32, [None, n_visible], name='x_input')
        lr = tf.placeholder(tf.float32)
        batch_size = tf.placeholder(tf.int32)
        return x, lr, batch_size

    def _create_variables(self, n_visible):
        W = tf.Variable(tf.random_normal([n_visible, self.n_hidden], stddev=0.01), name='W')
        Wuh = tf.Variable(tf.random_normal([self.n_hidden_recurrent, self.n_hidden], stddev=0.01), name='Wuh')
        Wux = tf.Variable(tf.random_normal([self.n_hidden_recurrent, n_visible], stddev=0.01), name='Wux')
        Wxu = tf.Variable(tf.random_normal([n_visible, self.n_hidden_recurrent], stddev=0.01), name='Wxu')
        Wuu = tf.Variable(tf.random_normal([self.n_hidden_recurrent, self.n_hidden_recurrent], stddev=0.01), name='Wuu')
        bu = tf.Variable(tf.zeros([1, self.n_hidden_recurrent]), name='bu')
        u0 = tf.Variable(tf.zeros([1, self.n_hidden_recurrent]), name='u0')
        bh = tf.Variable(tf.zeros([1, self.n_hidden]), name='bh')
        bx = tf.Variable(tf.zeros([1, n_visible]), name='bx')
        BH_t = tf.Variable(tf.zeros([1, self.n_hidden]), name='BH_t')
        BX_t = tf.Variable(tf.zeros([1, n_visible]), name='BX_t')

        return W, Wuh, Wux, Wxu, Wuu, bu, u0, bh, bx, BH_t, BX_t
