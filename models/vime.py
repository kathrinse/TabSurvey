import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from keras.layers import Input, Dense
from keras.models import Model
from keras import models
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers

from models.basemodel import BaseModel


class VIME(BaseModel):
    # Define all hyperparameters which are optimized
    PM = "p_m"
    ALPHA = "alpha"
    K = "K"
    BETA = "beta"

    def __init__(self, params, args):
        super().__init__(params, args)
        self.model_self = self.init_self(self.params[self.ALPHA])

        self.x_input = tf.placeholder(tf.float32, [None, args.dim])
        self.y_input = tf.placeholder(tf.float32, [None, args.num_classes])
        self.xu_input = tf.placeholder(tf.float32, [None, None, args.dim])

        # Build model
        y_hat_logit, self.y_hat = self.predictor(self.x_input)
        yv_hat_logit, yv_hat = self.predictor(self.xu_input)

        # Supervised loss
        self.y_loss = tf.losses.softmax_cross_entropy(self.y_input, y_hat_logit)
        # Unsupervised loss
        yu_loss = tf.reduce_mean(tf.nn.moments(yv_hat_logit, axes=0)[1])

        # Define variables
        p_vars = [v for v in tf.trainable_variables() if v.name.startswith('predictor')]

        # Define solver
        with tf.variable_scope('solver', reuse=tf.AUTO_REUSE):
            self.solver = tf.train.AdamOptimizer().minimize(self.y_loss + self.params[self.BETA] * yu_loss,
                                                            var_list=p_vars)

        # Start session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, y):
        x_unlab, x_label, y_label = split_data(X, y)

        encoder = self.train_self(x_unlab, self.params[self.PM])
        self.train_semi(encoder, x_label, y_label, x_unlab, self.params[self.PM], self.params[self.K])

    def predict(self, X):
        self.predictions = self.sess.run(self.y_hat, feed_dict={self.x_input: X})
        self.predictions = np.argmax(self.predictions, axis=1)

        # One Hot encode output
        self.predictions = pd.get_dummies(self.predictions)
        return self.predictions

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            cls.PM: trial.suggest_float(cls.PM, 0.1, 0.9),
            cls.ALPHA: trial.suggest_float(cls.ALPHA, 0.5, 3.0),
            cls.K: trial.suggest_int(cls.K, 2, 10),
            cls.BETA: trial.suggest_float(cls.BETA, 0.5, 3.0)
        }
        return params

    def init_self(self, alpha):
        dim = self.args.dim
        # Build model
        inputs = Input(shape=(dim,))
        # Encoder
        h = Dense(int(dim), activation='relu')(inputs)
        # Mask estimator
        output_1 = Dense(dim, activation='sigmoid', name='mask')(h)
        # Feature estimator
        output_2 = Dense(dim, activation='sigmoid', name='feature')(h)

        model = Model(inputs=inputs, outputs=[output_1, output_2])

        model.compile(optimizer='rmsprop',
                      loss={'mask': 'binary_crossentropy',
                            'feature': 'mean_squared_error'},
                      loss_weights={'mask': 1, 'feature': alpha})

        return model

    def train_self(self, x_unlab, p_m):
        # Generate corrupted samples
        m_unlab = mask_generator(p_m, x_unlab)
        m_label, x_tilde = pretext_generator(m_unlab, x_unlab)

        # Fit model on unlabeled data
        self.model_self.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, epochs=10, batch_size=128)

        # Extract encoder part
        layer_name = self.model_self.layers[1].name
        layer_output = self.model_self.get_layer(layer_name).output
        encoder = models.Model(inputs=self.model_self.input, outputs=layer_output)

        return encoder

    def predictor(self, x_input):
        hidden_dim = 100
        act_fn = tf.nn.relu

        with tf.variable_scope('predictor', reuse=tf.AUTO_REUSE):
            # Stacks multi-layered perceptron
            inter_layer = contrib_layers.fully_connected(x_input,
                                                         hidden_dim,
                                                         activation_fn=act_fn)
            inter_layer = contrib_layers.fully_connected(inter_layer,
                                                         hidden_dim,
                                                         activation_fn=act_fn)

            y_hat_logit = contrib_layers.fully_connected(inter_layer,
                                                         self.args.num_classes,
                                                         activation_fn=None)
            y_hat = tf.nn.softmax(y_hat_logit)

        return y_hat_logit, y_hat

    def train_semi(self, encoder, x_train, y_train, x_unlab, p_m, K):
        batch_size = 128
        iterations = 1000

        yv_loss_min_idx = -1

        # Training iteration loop
        for it in range(iterations):

            # Select a batch of labeled data
            batch_idx = np.random.permutation(len(x_train[:, 0]))[:batch_size]
            x_batch = x_train[batch_idx, :]
            y_batch = y_train[batch_idx, :]

            # Encode labeled data
            x_batch = encoder.predict(x_batch)

            # Select a batch of unlabeled data
            batch_u_idx = np.random.permutation(len(x_unlab[:, 0]))[:batch_size]
            xu_batch_ori = x_unlab[batch_u_idx, :]

            # Augment unlabeled data
            xu_batch = list()

            for rep in range(K):
                # Mask vector generation
                m_batch = mask_generator(p_m, xu_batch_ori)
                # Pretext generator
                _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)

                # Encode corrupted samples
                xu_batch_temp = encoder.predict(xu_batch_temp)
                xu_batch = xu_batch + [xu_batch_temp]
            # Convert list to matrix
            xu_batch = np.asarray(xu_batch)

            # Train the model
            _, y_loss_curr = self.sess.run([self.solver, self.y_loss],
                                           feed_dict={self.x_input: x_batch, self.y_input: y_batch,
                                                      self.xu_input: xu_batch})

            if yv_loss_min_idx + 100 < it:
                break


'''
    All VIME code copied: https://github.com/jsyoon0823/VIME
'''


def split_data(X, y):
    label_data_rate = 0.1

    # Divide labeled and unlabeled data
    idx = np.random.permutation(len(y))

    # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    # Unlabeled data
    x_unlab = X[unlab_idx, :]

    # Labeled data
    x_label = X[label_idx, :]
    y_label = y[label_idx, :]

    return x_unlab, x_label, y_label


def mask_generator(p_m, x):
    mask = np.random.binomial(1, p_m, x.shape)
    return mask


def pretext_generator(m, x):
    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def perf_metric(metric, y_test, y_test_hat):
    # Accuracy metric
    if metric == 'acc':
        result = accuracy_score(np.argmax(y_test, axis=1),
                                np.argmax(y_test_hat, axis=1))
    # AUROC metric
    elif metric == 'auc':
        result = roc_auc_score(y_test[:, 1], y_test_hat[:, 1])

    return result


def convert_matrix_to_vector(matrix):
    # Parameters
    no, dim = matrix.shape
    # Define output
    vector = np.zeros([no, ])

    # Convert matrix to vector
    for i in range(dim):
        idx = np.where(matrix[:, i] == 1)
        vector[idx] = i

    return vector


def convert_vector_to_matrix(vector):
    # Parameters
    no = len(vector)
    dim = len(np.unique(vector))
    # Define output
    matrix = np.zeros([no, dim])

    # Convert vector to matrix
    for i in range(dim):
        idx = np.where(vector == i)
        matrix[idx, i] = 1

    return matrix
