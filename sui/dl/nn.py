"""Product-based Neural Networks
https://arxiv.org/pdf/1611.00144.pdf
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


class PNN(tf.keras.Model):
    def __init__(self, features_dim: int, fields_dim: int, hidden_layer_sizes: list, dropout_params: list, product_layer_dim=10, lasso=0.01, ridge=1e-5, embedding_dim=10, product_type='ipnn'):
        super(PNN, self).__init__()
        self.features_dim = features_dim    # number of different
        self.fields_dim = fields_dim    # number of different original features
        self.dropout_params = dropout_params
        self.hidden_layer_sizes = hidden_layer_sizes    # number of hidden layers
        self.product_layer_dim = product_layer_dim
        self.lasso = lasso
        self.ridge = ridge
        self.embedding_dim = embedding_dim  # dimension of vectors after embedding
        self.product_type = product_type    # 'ipnn' for inner product while 'opnn' for outer product

        # embedding layer
        self.__init_embedding_layer()

        # product layer
        self.__init_product_layer()

        # hidden layers
        for layer_index in range(len(self.hidden_layer_sizes)):
            setattr(self, 'dense_' + str(layer_index), tf.keras.layers.Dense(self.hidden_layer_sizes[layer_index]))
            setattr(self, 'batch_norm_' + str(layer_index), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(layer_index), tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(layer_index), tf.keras.layers.Dropout(dropout_params[layer_index]))

        self.__create_model()

    def __init_embedding_layer(self):
        self.embedded_features = tf.keras.layers.Embedding(self.features_dim, self.embedding_dim, embeddings_initializer='uniform')

    def __init_product_layer(self):
        # linear signals l_z
        self.__init_linear_signals()
        # quadratic signals l_p
        self.__init_quadratic_signals()

    def __init_linear_signals(self, initializer=tf.initializers.GlorotUniform()):
        self.linear_signals = tf.Variable(initializer(shape=(self.product_layer_dim, self.fields_dim, self.embedding_dim)))

    def __init_quadratic_signals(self, initializer=tf.initializers.GlorotUniform()):
        assert self.product_type in ['ipnn', 'opnn'], "'product_type' should be either 'ipnn' or 'opnn'."
        if self.product_type == 'ipnn':
            self.theta = tf.Variable(initializer(shape=(self.product_layer_dim, self.fields_dim)))
        elif self.product_type == 'opnn':
            pass
        else:
            raise Exception('Arcane exception...')

    def __create_model(self):

        model = tf.keras.Sequential()

        # the product layer
        # # linear signals l_z
        # # quadratic signals l_p
        # the first hidden layer

        # 使用双层卷积结构的方式提取图像特征，第一层双层卷积使用使用3*3的卷积核，输出维度是64，全局使用he_normal进行kernel_initializer，激活函数使用relu
        model.add(
            tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   input_shape=(32, 32, 3), name="conv1"))
        model.add(
            tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv2"))
        # 使用tf.keras.layers.MaxPool2D搭建神经网络的池化层，使用最大值池化策略，将2*2局域的像素使用一个最大值代替，步幅为2，padding使用valid策略
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool1"))
        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(Dropout(rate=self.rate, name="d1"))
        # 使用batchnormalization对上一层的输出数据进行归一化
        model.add(BatchNormalization())
        # 使用双层卷积结构的方式提取图像特征，第二层双层卷积使用使用3*3的卷积核，输出维度是128，全局使用he_normal进行kernel_initializer,激活函数使用relu
        model.add(
            tf.keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv3"))
        model.add(
            tf.keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv4"))
        # 使用tf.keras.layers.MaxPool2D搭建神经网络的池化层，使用最大值池化策略，将2*2局域的像素使用一个最大值代替，步幅为2，padding使用valid策略
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool2"))
        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(Dropout(rate=self.rate, name="d2"))
        # 使用batchnormalization对上一层的输出数据进行归一化
        model.add(tf.keras.layers.BatchNormalization())
        # 使用双层卷积结构的方式提取图像特征，第三层双层卷积使用使用3*3的卷积核，输出维度是128，全局使用he_normal进行kernel_initializer,激活函数使用relu
        model.add(
            tf.keras.layers.Conv2D(256, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv5"))
        model.add(
            tf.keras.layers.Conv2D(256, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv6"))
        # 使用tf.keras.layers.MaxPool2D搭建神经网络的池化层，使用最大值池化策略，将2*2局域的像素使用一个最大值代替，步幅为2，padding使用valid策略
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool3"))
        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(Dropout(rate=self.rate, name="d3"))
        # 使用batchnormalization对上一层的输出数据进行归一化
        model.add(BatchNormalization())
        # 使用flatten将上层的输出数据压平
        model.add(Flatten(name="flatten"))
        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(Dropout(self.rate))
        # 叠加一层全连接层，用于拟合最终结果，激活函数使用relu
        model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        # 叠加一层Dropout层，提高泛化性，降低神经网络的复杂度
        model.add(Dropout(self.rate))
        # 叠加一层全连接层作为l1层，激活函数使用relu
        model.add(Dense(2, activation='relu', kernel_initializer='he_normal', name='l1'))
        # 叠加一层全连接层作为l2层，激活函数使用relu
        model.add(Dense(2, activation='relu', kernel_initializer='he_normal', name='l2'))
        # 叠加一层全连接层作为输出层，激活函数使用sigmoid
        model.add(Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='output'))
        # 最后编译搭建完成的神经网络，使用categorical_crossentropy损失函数，adam优化器，模型的衡量指标是准确率
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self):
        pass

    def predict(self):
        pass

    def dump(self):
        pass

    def restore(self):
        pass
