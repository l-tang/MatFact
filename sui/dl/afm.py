"""Attentional Factorization Machines
https://arxiv.org/abs/1708.04617
Date: 17/Aug/2020
Author: Li Tang
"""
import tensorflow as tf

import pickle
# from util.train_model import train_test_model_demo

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class SuiAFMError(Exception):
    pass


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size=10, attention_size=3, **kwargs):
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]

        self.linearlayer = tf.keras.layers.Dense(input_dim, activation='relu', use_bias=True)
        self.attention_w = self.add_weight(shape=(self.embedding_size, self.attention_size),
                                           initializer='random_normal', trainable=True)
        self.attention_b = self.add_weight(shape=(self.attention_size,),
                                           initializer='random_normal', trainable=True)
        self.attention_h = self.add_weight(shape=(self.attention_size,),
                                           initializer='random_normal', trainable=True)
        self.attention_p = self.add_weight(shape=(self.embedding_size, 1),
                                           initializer='ones', trainable=True)

    def call(self, input):
        # element_wise
        feature_dim = input.shape[1]
        element_wise_product_list = []
        for i in range(feature_dim):
            for j in range(i + 1, feature_dim):
                element_wise_product_list.append(tf.multiply(input[:, i, :], input[:, j, :]))  # None * embedding_size
        self.element_wise_product = tf.stack(element_wise_product_list)  # (F * F - 1 / 2) * None * embedding_size
        self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2],
                                                 name='element_wise_product')  # None * (F * F - 1 / 2) *  embedding_size
        print("element_wise_product", self.element_wise_product.get_shape())
        # attention part
        num_interaction = int(feature_dim * (feature_dim - 1) / 2)
        # wx+b->relu(wx+b)->h*relu(wx+b)
        self.attention_wx_plus_b = tf.reshape(
            tf.add(tf.matmul(tf.reshape(self.element_wise_product, shape=(-1, self.embedding_size)),
                             self.attention_w), self.attention_b),
            shape=[-1, num_interaction, self.attention_size])  # N *(F*F-1/2)*1
        self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(tf.nn.relu(self.attention_wx_plus_b),
                                                              self.attention_h), axis=2))  # N * ( F * F - 1 / 2) * 1

        self.attention_exp_sum = tf.reshape(tf.reduce_sum(self.attention_exp, axis=1), shape=(-1, 1))  # N * 1 * 1

        self.attention_out = tf.divide(self.attention_exp, self.attention_exp_sum,
                                       name='attention_out')  # N * ( F * F - 1 / 2) * 1
        self.attention_x_product = tf.reduce_sum(
            tf.einsum('bn,bnm->bnm', self.attention_out, self.element_wise_product), axis=1,
            name='afm')  # N * embedding_size
        self.attention_part_sum = tf.matmul(self.attention_x_product, self.attention_p)  # N * 1

        return self.attention_part_sum


class AFM(tf.keras.Model):
    def __init__(self, feature_dim, embedding_size=10, attention_size=3):
        super().__init__()
        self.feature_dim = feature_dim  # F features nums 字典数量
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        # Embedding 这里采用embeddings层 因此大小为F* M F为field特征数量，N 为 feature的种类数 M为embedding的维度
        feat_embeddings = tf.keras.layers.Embedding(feature_dim, embedding_size,
                                                    embeddings_initializer='uniform')  # N * embedding_size
        self.feat_embeddings = feat_embeddings
        self.attentionlayer = AttentionLayer(self.embedding_size, self.attention_size)
        # linear part
        self.linearlayer = tf.keras.layers.Dense(1, activation='relu', use_bias=True)

    def call(self, feat_index, feat_value):
        # call函数接收输入变量
        # embedding part  feat_index = inputs为输入 feat_embeddings为一个layer。
        feat_embedding_0 = self.feat_embeddings(feat_index)  # Batch * F * embedding_size
        feat_embedding = tf.einsum('bnm,bn->bnm', feat_embedding_0, feat_value)  # # Batch * F * embedding_size
        feat_embedding_1 = tf.transpose(feat_embedding, perm=[0, 2, 1])
        y_deep = self.attentionlayer(feat_embedding)

        y_linear = tf.reduce_sum(self.linearlayer(feat_embedding_1), axis=1)
        output = y_deep + y_linear
        return output
