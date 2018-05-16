import tensorflow as tf
import numpy as np
import tqdm


class RNN_EBM(object):
    """ Recurrent Energy-Based Model implementation using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, num_epochs=100, batch_size=10, n_hidden=50, n_hidden_recurrent=100):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden  # Size of RBM's hidden layer
        self.n_hidden_recurrent = n_hidden_recurrent  # Size of RNN's hidden layer

        # Placeholders
        self.input_data = None
        self.lr = None

        # Variables
        self.W, self.Wuh, self.Wux, self.Wxu, self.Wuu, self.bu, \
            self.u0, self.bh, self.bx, self.BH_t, self.BX_t = \
            None, None, None, None, None, None, None, None, None, None, None
        self.tvars = []

        self.update = None
        self.cost = None

        self.tf_session = None
        self.tf_saver = None

    def fit(self, X):
        self._build_model(X.shape[0])

        self.tf_session = tf.Session()
        self._initialize_tf_utilities_and_ops()
        self._train_model(X)

    def predict(self, X, min_energy=None):
        results = []
        scores = []

        for i in range(len(X)):
          feed = {
            self.input_data: X.iloc[i]
          }
          reconstruction_err = sess.run([cost], feed_dict=feed)
          results += [reconstruction_err]
          if min_energy is not None:
            scores += [reconstruction_err >= min_energy]

        return scores, results if min_energy is not None else results

    def _train_model(self, train_set):
        for epoch in tqdm(range(self.num_epochs)):
            costs = []
            start = time.time()
            for i in range(0, len(train_set), self.batch_size):
                x = train_set[i:i + self.batch_size]
                alpha = min(0.01, 0.1 / float(i + 1))

                _, C = sess.run([self.update, self.cost], feed_dict={self.input_data: tr_x, self.lr: alpha})
                costs.append(C)
            print(f'Epoch: {epoch} Cost: {np.mean(costs)} Time: {time.time() - start}')

    def _initialize_tf(self):
        init = tf.initialize_all_variables()
        self.tf_session.run(init)

    def _build_model(self, n_visible):
        self.input_data, self.lr = self._create_placeholders(n_visible)
        self.W, self.Wuh, self.Wux, self.Wxu, self.Wuu, self.bu, \
            self.u0, self.bh, self.bx, self.BH_t, self.BX_t = self._create_variables(n_visible)

        def rnn_recurrence(u_tmin1, sl):
            # Iterate through the data in the batch and generate the values of the RNN hidden nodes
            sl = tf.reshape(sl, [1, n_visible])
            u_t = tf.nn.softplus(bu + tf.matmul(sl, Wxu) + tf.matmul(u_tmin1, Wuu))
            return u_t

        def visible_bias_recurrence(bx_t, u_tmin1):
            # Iterate through the values of the RNN hidden nodes and generate the values of the visible bias vectors
            bx_t = tf.add(bx, tf.matmul(u_tmin1, Wux))
            return bx_t

        def hidden_bias_recurrence(bh_t, u_tmin1):
            # Iterate through the values of the RNN hidden nodes and generate the values of the hidden bias vectors
            bh_t = tf.add(bh, tf.matmul(u_tmin1, Wuh))
            return bh_t

        tf.assign(BH_t, tf.tile(BH_t, [size_bt, 1]))
        tf.assign(BX_t, tf.tile(BX_t, [size_bt, 1]))

        # Scan through the rnn and generate the value for each hidden node in the batch
        u_t = tf.scan(rnn_recurrence, x, initializer=u0)
        # Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
        BX_t = tf.reshape(tf.scan(visible_bias_recurrence, u_t, tf.zeros([1, n_visible], tf.float32)),
                          [n_visible, size_bt])
        BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, u_t, tf.zeros([1, n_hidden], tf.float32)),
                          [n_hidden, size_bt])

        self.cost = self._run_ebm(x, W, BX_t, BH_t)

        self.tvars = [self.W, self.Wuh, self.Wux, self.Wxu, self.Wuu, self.bu, self.u0, self.bh, self.bx]
        opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars), 1)
        self.update = opt_func.apply_gradients(zip(grads, self.tvars))

    def _create_placeholders(self, n_visible):
        x = tf.placeholder(tf.float32, [None, n_visible], name='x_input')
        lr = tf.placeholder(tf.float32)
        return x, lr

    def _create_variables(self, n_visible):
        W = tf.Variable(tf.random_normal([n_visible, self.n_hidden], stddev=0.01), name="W")
        Wuh = tf.Variable(tf.random_normal([self.n_hidden_recurrent, self.n_hidden], stddev=0.01), name="Wuh")
        Wux = tf.Variable(tf.random_normal([self.n_hidden_recurrent, n_visible], stddev=0.01), name="Wux")
        Wxu = tf.Variable(tf.random_normal([n_visible, self.n_hidden_recurrent], stddev=0.01), name="Wxu")
        Wuu = tf.Variable(tf.random_normal([self.n_hidden_recurrent, self.n_hidden_recurrent], stddev=0.01), name="Wuu")
        bu = tf.Variable(tf.zeros([1, self.n_hidden_recurrent]), name="bu")
        u0 = tf.Variable(tf.zeros([1, self.n_hidden_recurrent]), name="u0")
        bh = tf.Variable(tf.zeros([1, self.n_hidden]), name="bh")
        bx = tf.Variable(tf.zeros([1, n_visible]), name="bx")
        BH_t = tf.Variable(tf.zeros([1, self.n_hidden]), name="BH_t")
        BX_t = tf.Variable(tf.zeros([1, n_visible]), name="BX_t")

        return W, Wuh, Wux, Wxu, Wuu, bu, u0, bh, bx, BH_t, BX_t

    def run_ebm(x, W, b_prime, b):
        """ Runs EBM for time step and returns reconstruction error.
        1-layer implementation, TODO: implement and test deep structure
        """
        x = tf.transpose(x)  # For batch processing

        forward = tf.matmul(tf.transpose(W), x) + b_t
        reconstruction = tf.matmul(W, tf.sigmoid(forward)) + b_prime_t
        loss = tf.reduce_sum(tf.square(x - reconstruction))
        return loss

    def delete(self):
        self.tf_session.close()
