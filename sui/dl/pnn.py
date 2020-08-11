"""Product-based Neural Networks
https://arxiv.org/pdf/1611.00144.pdf
Date: 14/Jul/2020
Author: Li Tang
"""
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout

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
                 product_layer_dim=10, lasso=0.01, ridge=1e-5, embedding_dim=10, product_type='ipnn',
                 initializer=tf.initializers.GlorotUniform()):
        super().__init__()
        self.features_dim = features_dim  # number of different, denoted by F
        self.fields_dim = fields_dim  # number of different original features
        self.dropout_params = dropout_params
        self.hidden_layer_sizes = hidden_layer_sizes  # number of hidden layers
        self.product_layer_dim = product_layer_dim
        self.lasso = lasso
        self.ridge = ridge
        self.embedding_dim = embedding_dim  # dimension of vectors after embedding, denoted by M
        self.product_type = product_type  # 'ipnn' for inner product while 'opnn' for outer product
        self.initializer = initializer

        # embedding layer
        # the size of embedding layer is F * M
        self.embedding_layer = tf.keras.layers.Embedding(self.features_dim, self.embedding_dim,
                                                         embeddings_initializer='uniform')

        # product layer
        # linear signals l_z
        self.linear_sigals_variable = tf.Variable(
            self.initializer(shape=(self.product_layer_dim, self.fields_dim, self.embedding_dim)))
        # quadratic signals l_p
        self.quadratic_signals_variable = self.__init_quadratic_signals()

        # hidden layers
        self.__init_hidden_layers()

        # output layer
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)

    def __init_quadratic_signals(self, initializer=tf.initializers.GlorotUniform()):
        if self.product_type == 'ipnn':
            # matrix decomposition based on the assumption: W_p^n = \theta ^n * {\theta^n}^T
            return tf.Variable(initializer(shape=(self.product_layer_dim, self.fields_dim)))
        elif self.product_type == 'opnn':
            # TODO
            pass
        else:
            raise SuiPNNError("'product_type' should be either 'ipnn' or 'opnn'.")

    def __init_hidden_layers(self):
        for layer_index in range(len(self.hidden_layer_sizes)):
            setattr(self, 'dense_' + str(layer_index), Dense(self.hidden_layer_sizes[layer_index]))
            setattr(self, 'batch_norm_' + str(layer_index), BatchNormalization())
            setattr(self, 'activation_' + str(layer_index), Activation('relu'))
            setattr(self, 'dropout_' + str(layer_index), Dropout(self.dropout_params[layer_index]))

    def call(self, feature_index, feature_value, use_dropout=True):
        # linear part
        l_z = tf.einsum('bnm,dnm->bd', self.embedding_layer(feature_index), self.linear_sigals_variable)  # Batch * D1

        # quadratic part
        if self.product_type == 'ipnn':
            theta = tf.einsum('bnm,dn->bdnm', self.embedding_layer(feature_index),
                              self.quadratic_signals_variable)  # Batch * D1 * N * M
            l_p = tf.einsum('bdnm,bdnm->bd', theta, theta)
        else:
            # TODO
            pass

        model = tf.concat((l_z, l_p), axis=1)
        if use_dropout:
            model = tf.keras.layers.Dropout(self.dropout_params[0])(model)

        for i in range(len(self.hidden_layer_sizes)):
            model = getattr(self, 'dense_' + str(i))(model)
            model = getattr(self, 'batch_norm_' + str(i))(model)
            model = getattr(self, 'activation_' + str(i))(model)
            if use_dropout:
                model = getattr(self, 'dropout_' + str(i))(model)

        output = self.output_layer(model)
        return output

    # TODO
    def dump(self):
        pass

    # TODO
    @staticmethod
    def restore():
        return None
