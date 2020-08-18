"""Module to recall based on recurrent neural networks.
Date: 14/Jul/2020
Author: Li Tang
"""
__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.1.9'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'

import tensorflow as tf
from tensorflow.keras.layers import GRU


class RNN4Rec:
    def __init__(self):
        pass


class GRU4Rec(GRU):
    def __init__(self, units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                 recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False,
                 go_backwards=False, stateful=False, unroll=False, time_major=False, reset_after=True, **kwargs):
        super().__init__(units, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer,
                         bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                         recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout,
                         recurrent_dropout=recurrent_dropout, implementation=implementation,
                         return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards,
                         stateful=stateful, unroll=unroll, time_major=time_major, reset_after=reset_after, **kwargs)


class BERT4Rec(RNN4Rec):
    def __init__(self):
        super().__init__()
