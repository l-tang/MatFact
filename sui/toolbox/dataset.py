"""
    Module to load data to test models
    Date: 15/May/2020
    Author: Li Tang
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import os

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.1.6'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


def iris():
    data_path = os.path.split(os.path.realpath(__file__))[0] + '/data/iris.csv'
    return pd.read_csv(data_path, header=None)


def titanic():
    data_path = os.path.split(os.path.realpath(__file__))[0] + '/data/titanic.csv'
    return pd.read_csv(data_path)


def movielens_1m(target='ratings'):
    data_path = os.path.split(os.path.realpath(__file__))[0] + '/data/movielens_1m/{}.csv'.format(target)
    return pd.read_csv(data_path)
