"""Product-based Neural Networks
https://arxiv.org/pdf/1611.00144.pdf
Date: 14/Jul/2020
Author: Li Tang
"""
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
import numpy as np
from .initializers import get_init
from .losses import get_loss
from .optimizers import get_opti

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class SuiPNNError(Exception):
    pass


class PNN(tf.keras.Model):
    def __init__(self, features_dim: int, fields_dim: int, hidden_layer_sizes: list, dropout_params: list,
                 product_layer_dim=10, lasso=0.01, ridge=1e-5, embedding_dim=10, product_type='pnn',
                 initializer='glorotuniform', activation='sigmoid', hidden_activation='relu'):
        super().__init__()
        self.features_dim = features_dim  # size of features after one-hot, denoted by F
        self.fields_dim = fields_dim  # number of different original features, denoted by N
        self.dropout_params = dropout_params
        self.hidden_layer_sizes = hidden_layer_sizes  # number of hidden layers
        self.product_layer_dim = product_layer_dim  # as same as the input size of l_1, denoted by D_1
        self.lasso = lasso
        self.ridge = ridge
        self.embedding_dim = embedding_dim  # dimension of vectors after embedding, denoted by M
        # product type for product layer
        # 'ipnn' for inner product , 'opnn' for outer product, and 'pnn' for concatenating both product
        self.product_type = product_type
        self.initializer = get_init(initializer)
        self.activation = activation
        self.hidden_activation = hidden_activation

        # embedding layer
        # the size of embedding layer is F * M
        self.embedding_layer = tf.keras.layers.Embedding(self.features_dim, self.embedding_dim,
                                                         embeddings_initializer='uniform')

        # product layer
        # linear signals l_z
        self.linear_sigals_variable = tf.Variable(
            self.initializer(shape=(self.product_layer_dim, self.fields_dim, self.embedding_dim)))
        # quadratic signals l_p
        self.__init_quadratic_signals()

        # hidden layers
        self.__init_hidden_layers()

        # output layer
        self.output_layer = tf.keras.layers.Dense(1, activation=self.activation, use_bias=True)

    def __init_quadratic_signals(self):
        if self.product_type == 'ipnn':
            # matrix decomposition based on the assumption: W_p^n = \theta ^n * {\theta^n}^T
            # then the size of W_p^n is D_1 * N
            self.theta = tf.Variable(self.initializer(shape=(self.product_layer_dim, self.fields_dim)))
        elif self.product_type == 'opnn':
            # the size of W_p^n is D_1 * M * M
            self.quadratic_weights = tf.Variable(
                self.initializer(shape=(self.product_layer_dim, self.embedding_dim, self.embedding_dim)))
        elif self.product_type == 'pnn':
            self.theta = tf.Variable(self.initializer(shape=(self.product_layer_dim, self.fields_dim)))
            self.quadratic_weights = tf.Variable(
                self.initializer(shape=(self.product_layer_dim, self.embedding_dim, self.embedding_dim)))
        else:
            raise SuiPNNError("'product_type' should be 'ipnn', 'opnn', or 'pnn'.")

    def __init_hidden_layers(self):
        for layer_index in range(len(self.hidden_layer_sizes)):
            setattr(self, 'dense_' + str(layer_index), Dense(self.hidden_layer_sizes[layer_index]))
            setattr(self, 'batch_norm_' + str(layer_index), BatchNormalization())
            setattr(self, 'activation_' + str(layer_index), Activation(self.hidden_activation))
            setattr(self, 'dropout_' + str(layer_index), Dropout(self.dropout_params[layer_index]))

    def call(self, feature_value, embedding_index, training=False):
        features = tf.einsum('bnm,bn->bnm', self.embedding_layer(embedding_index), feature_value)
        # linear part
        l_z = tf.einsum('bnm,dnm->bd', features, self.linear_sigals_variable)  # Batch * D_1

        # quadratic part
        if self.product_type == 'ipnn':
            delta = tf.einsum('dn,bnm->bdnm', self.theta, features)  # Batch * D_1 * N * M
            l_p = tf.einsum('bdnm,bdnm->bd', delta, delta)
        elif self.product_type == 'opnn':
            sum_features = tf.einsum('bnm->bm', features)  # Batch * M
            p = tf.einsum('bm,bn->bmn', sum_features, sum_features)
            l_p = tf.einsum('bmn,dmn->bd', p, self.quadratic_weights)
        elif self.product_type == 'pnn':
            delta = tf.einsum('dn,bnm->bdnm', self.theta, features)  # Batch * D_1 * N * M
            sum_features = tf.einsum('bnm->bm', features)  # Batch * M
            p = tf.einsum('bm,bn->bmn', sum_features, sum_features)
            l_p = tf.concat(
                (tf.einsum('bdnm,bdnm->bd', delta, delta), tf.einsum('bmn,dmn->bd', p, self.quadratic_weights)), axis=1)
        else:
            raise SuiPNNError("'product_type' should be 'ipnn', 'opnn', or 'pnn'.")

        model = tf.concat((l_z, l_p), axis=1)
        if training:
            model = tf.keras.layers.Dropout(self.dropout_params[0])(model)

        for i in range(len(self.hidden_layer_sizes)):
            model = getattr(self, 'dense_' + str(i))(model)
            model = getattr(self, 'batch_norm_' + str(i))(model)
            model = getattr(self, 'activation_' + str(i))(model)
            if training:
                model = getattr(self, 'dropout_' + str(i))(model)

        return self.output_layer(model)

    def train(self, feature_value, embedding_index, label, optimizer='adam', learning_rate=1e-4, loss='sigmoid',
              epochs=50, batch=32, shuffle=10000):
        for epoch in range(epochs):
            train_set = tf.data.Dataset.from_tensor_slices((feature_value, embedding_index, label)).shuffle(
                shuffle).batch(batch, drop_remainder=True)
            for batch_set in train_set:
                with tf.GradientTape() as tape:
                    prediction = self.call(feature_value=batch_set[0], embedding_index=batch_set[1], training=True)
                    self.loss_obj = get_loss(loss)
                    self.optimizer = get_opti(optimizer)(learning_rate=learning_rate)
                    batch_loss = self.loss_obj(batch_set[2], prediction)
                gradients = tape.gradient(batch_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            mean_loss = tf.keras.metrics.Mean(name='train_loss')
            print('epoch: {} ==> loss: {}'.format(epoch + 1, mean_loss(batch_loss)))

    def predict(self, feature_value, embedding_index):
        feature_value = tf.convert_to_tensor(feature_value)
        embedding_index = tf.convert_to_tensor(embedding_index)
        return self.call(feature_value=feature_value, embedding_index=embedding_index, training=False)

    # TODO
    def dump(self, path):
        self.save(filepath=path)

    # TODO
    @staticmethod
    def restore():
        return None
