"""Module including some useful implementations apropos neural networks.
Date: 14/Jul/2020
Author: Li Tang
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from sklearn.preprocessing import OneHotEncoder

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.1.9'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


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


# if __name__ == '__main__':
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

import pickle
import tensorflow as tf
from train_model_util_TensorFlow import train_test_model_demo

AID_DATA_DIR = '../data/Criteo/forOtherModels/'  # 辅助用途的文件路径
AID_DATA_DIR = './'

"""
TensorFlow 2.0 implementation of Product-based Neural Network[1]
Reference:
[1] Product-based Neural Networks for User ResponsePrediction,
    Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, Jun Wang
[2] Tensorflow implementation of PNN
    https://github.com/Atomu2014/product-nets
"""


class PNN_layer(tf.keras.Model):

    def __init__(self, num_feat, num_field, dropout_deep, deep_layer_sizes, product_layer_dim=10,
                 reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer'):
        super().__init__()  # Python2 下使用 super(PNN_layer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.num_feat = num_feat  # Denoted as
        self.num_field = num_field  # Denoted as N
        self.product_layer_dim = product_layer_dim  # Denoted as D1
        self.dropout_deep = dropout_deep

        # Embedding
        feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size, embeddings_initializer='uniform')
        self.feat_embeddings = feat_embeddings

        initializer = tf.initializers.GlorotUniform()

        # linear part
        self.linear_weights = tf.Variable(initializer(shape=(product_layer_dim, num_field, embedding_size)))

        # quadratic part
        self.product_type = product_type
        if product_type == 'inner':
            self.theta = tf.Variable(initializer(shape=(product_layer_dim, num_field)))  # D1 * N
        else:
            self.quadratic_weights = tf.Variable(initializer(shape=(product_layer_dim, embedding_size,
                                                                    embedding_size)))  # D1 * M * M

        # fc layer
        self.deep_layer_sizes = deep_layer_sizes

        # 神经网络方面的参数
        for i in range(len(deep_layer_sizes)):
            setattr(self, 'dense_' + str(i), tf.keras.layers.Dense(deep_layer_sizes[i]))
            setattr(self, 'batchNorm_' + str(i), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i), tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(i), tf.keras.layers.Dropout(dropout_deep[i]))

        # last layer
        self.fc = tf.keras.layers.Dense(1, activation=None, use_bias=True)

    def call(self, feat_index, feat_value, use_dropout=True):
        # embedding part
        feat_embedding = self.feat_embeddings(feat_index)  # Batch * N * M

        # linear part
        lz = tf.einsum('bnm,dnm->bd', feat_embedding, self.linear_weights)  # Batch * D1

        # quadratic part
        if self.product_type == 'inner':
            theta = tf.einsum('bnm,dn->bdnm', feat_embedding, self.theta)  # Batch * D1 * N * M
            lp = tf.einsum('bdnm,bdnm->bd', theta, theta)
        else:
            embed_sum = tf.reduce_sum(feat_embedding, axis=1)
            p = tf.einsum('bm,bn->bmn', embed_sum, embed_sum)
            lp = tf.einsum('bmn,dmn->bd', p, self.quadratic_weights)  # Batch * D1

        y_deep = tf.concat((lz, lp), axis=1)
        if use_dropout:
            y_deep = tf.keras.layers.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(len(self.deep_layer_sizes)):
            y_deep = getattr(self, 'dense_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = getattr(self, 'activation_' + str(i))(y_deep)
            if use_dropout:
                y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        output = self.fc(y_deep)
        return output


if __name__ == '__main__':
    feat_dict_ = pickle.load(open(AID_DATA_DIR + 'aid_data/feat_dict_10.pkl2', 'rb'))

    pnn = PNN_layer(num_feat=len(feat_dict_) + 1, num_field=39, dropout_deep=[0.5, 0.5, 0.5],
                    deep_layer_sizes=[400, 400], product_layer_dim=10,
                    reg_l1=0.01, reg_l2=1e-5, embedding_size=10, product_type='outer')

    train_label_path = AID_DATA_DIR + 'train_label'
    train_idx_path = AID_DATA_DIR + 'train_idx'
    train_value_path = AID_DATA_DIR + 'train_value'

    test_label_path = AID_DATA_DIR + 'test_label'
    test_idx_path = AID_DATA_DIR + 'test_idx'
    test_value_path = AID_DATA_DIR + 'test_value'

    train_test_model_demo(pnn, train_label_path, train_idx_path, train_value_path, test_label_path, test_idx_path,
                          test_value_path)