"""Module including some useful implementations apropos neural networks.
"""
__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'GPL'
__version__ = '1.0.1'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from sklearn.preprocessing import OneHotEncoder
from sui.ml.mf import FunkSVD
import sui
from sui import ml
from sui import rl
from sui import dl
from sui import dataset


class PNN(tf.keras.Model):
    """Product-based Neural Networks
    https://arxiv.org/pdf/1611.00144.pdf
    """

    def __init__(self, features_dim: int, fields_dim: int, hidden_layer_sizes: list, dropout_params: list,
                 product_layer_dim=10, lasso=0.01, ridge=1e-5, embedding_dim=10, product_type='ipnn', model=None):
        super(PNN, self).__init__()
        self.features_dim = features_dim  # number of different, denoted by F
        self.fields_dim = fields_dim  # number of different original features
        self.dropout_params = dropout_params
        self.hidden_layer_sizes = hidden_layer_sizes  # number of hidden layers
        self.product_layer_dim = product_layer_dim
        self.lasso = lasso
        self.ridge = ridge
        self.embedding_dim = embedding_dim  # dimension of vectors after embedding, denoted by M
        self.product_type = product_type  # 'ipnn' for inner product while 'opnn' for outer product

        self.model = model
        # if there is no pre-trained model to use
        if self.model is None:
            # embedding layer
            self.embedding_layer = self.__init_embedding_layer()

            # product layer
            self.product_layer = self.__init_product_layer()

            # hidden layers
            for layer_index in range(len(self.hidden_layer_sizes)):
                setattr(self, 'dense_' + str(layer_index), tf.keras.layers.Dense(self.hidden_layer_sizes[layer_index]))
                setattr(self, 'batch_norm_' + str(layer_index), tf.keras.layers.BatchNormalization())
                setattr(self, 'activation_' + str(layer_index), tf.keras.layers.Activation('relu'))
                setattr(self, 'dropout_' + str(layer_index), tf.keras.layers.Dropout(dropout_params[layer_index]))

    def __init_embedding_layer(self):
        # the size of embedding layer is F * M
        return tf.keras.layers.Embedding(self.features_dim, self.embedding_dim, embeddings_initializer='uniform')

    def __init_product_layer(self):
        # linear signals l_z
        self.__linear_sigals_variable = self.__init_linear_signals()
        # quadratic signals l_p
        self.__quadratic_signals_variable = self.__init_quadratic_signals()
        return Dense(self.product_layer_dim, activation='relu', kernel_initializer='he_normal', name='l1')

    def __init_linear_signals(self, initializer=tf.initializers.GlorotUniform()):
        return tf.Variable(initializer(shape=(self.product_layer_dim, self.fields_dim, self.embedding_dim)))

    def __init_quadratic_signals(self, initializer=tf.initializers.GlorotUniform()):
        assert self.product_type in ['ipnn', 'opnn'], "'product_type' should be either 'ipnn' or 'opnn'."
        if self.product_type == 'ipnn':
            # matrix decomposition based on the assumption: W_p^n = \theta ^n * {\theta^n}^T
            return tf.Variable(initializer(shape=(self.product_layer_dim, self.fields_dim)))
        elif self.product_type == 'opnn':
            pass
        else:
            raise Exception('Arcane exception...')

    def loss_function(self, labels, logits, name=None):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits, name=name
        )
        return loss

    def __create_model(self):
        model = tf.keras.Sequential()
        # feat_embedding_0 = self.embedding_layer(feat_index)  # Batch * N * M
        # feat_embedding = tf.einsum('bnm,bn->bnm', feat_embedding_0, feat_value)
        # tf.einsum('bnm,dnm->bd', feat_embedding, self.linear_signals)
        model.add(self.product_layer)
        model.add(Dense(2, activation='relu', kernel_initializer='he_normal', name='l1'))
        # 叠加一层全连接层作为l2层，激活函数使用relu
        model.add(Dense(2, activation='relu', kernel_initializer='he_normal', name='l2'))
        # 叠加一层全连接层作为输出层，激活函数使用sigmoid
        model.add(Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='output'))
        # 最后编译搭建完成的神经网络，使用categorical_crossentropy损失函数，adam优化器，模型的衡量指标是准确率
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def train(self, data, labels, batch_size=None, epochs=1, verbose=1):
        if self.model is None:
            self.__create_model()
        self.model.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, data, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1,
                use_multiprocessing=False):
        return self.model.predict(x=data, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks,
                                  max_queue_size=max_queue_size, workers=workers,
                                  use_multiprocessing=use_multiprocessing)

    def dump(self):
        pass

    def restore(self):
        pass


if __name__ == '__main__':
    # pnn = PNN(features_dim=4, fields_dim=3, hidden_layer_sizes=[64, 32, 8], dropout_params=[0.5] * 3)
    # # path = '../../data/titanic.csv'
    # # import pandas as pd
    # # import numpy as np
    # #
    # # data = pd.read_csv(path)
    # # data.fillna(0)
    # # data, labels = data[data.columns.difference(['Survived'])], data['Survived']
    # # print(data[data.columns.difference(['Age', 'Fare'])])
    # # # print(pd.Categorical(data[data.columns.difference(['Age', 'Fare'])]).codes.astype(int))
    # # data[data.columns.difference(['Age', 'Fare'])] = pd.Categorical(
    # #     data[data.columns.difference(['Age', 'Fare'])]).codes.astype(int)
    # # for i in np.where(np.isnan(data))[0]:
    # #     data['Age'][i] = 0
    # # # print(data[data.columns.difference(['Age', 'Fare'])])
    # # one_hot_encoder = OneHotEncoder().fit(data[data.columns.difference(['Age', 'Fare'])])
    # # # print(one_hot_encoder.transform(data[data.columns.difference(['Age', 'Fare'])]).toarray())
    # #
    # # # print(one_hot_encoder.transform(data).toarray())
    # # # pnn.train(data, labels)
    # print(pnn.features_dim,
    #       pnn.fields_dim,
    #       pnn.dropout_params,
    #       pnn.hidden_layer_sizes,
    #       pnn.product_layer_dim,
    #       pnn.lasso,
    #       pnn.ridge,
    #       pnn.embedding_dim,
    #       pnn.product_type,
    #       pnn.model,
    #       pnn.dense_2)
    # pnn.train(data=[(0, 1, 0, 1),
    #                 (0, 1, 1, 1),
    #                 (1, 0, 1, 0),
    #                 (0, 1, 0, 2)], labels=[0, 2, 76, 3], epochs=10)
    # print(pnn.predict([(0, 1, 0, 1)]))
    print(dl.model_class_list)
    print(ml.model_class_list)
    print(rl.model_class_list)
    print(dataset.iris())
