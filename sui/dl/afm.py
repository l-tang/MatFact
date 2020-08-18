"""Attentional Factorization Machines
https://arxiv.org/abs/1708.04617
Date: 17/Aug/2020
Author: Li Tang
"""
import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from .initializers import get_init

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


class AFM:
    def __init__(self):
        pass

    def call(self):
        pass
