"""
    Module to load data to test models
    Date: 15/May/2020
    Author: Li Tang
"""
__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'GPL'
__version__ = '1.0.1'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'

import pandas as pd
import os


def iris():
    data_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/iris.csv'
    return pd.read_csv(data_path, header=None)


def titanic():
    data_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/titanic.csv'
    return pd.read_csv(data_path)


def movielens_1m(target='ratings'):
    data_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/movielens_1m/{}.csv'.format(target)
    return pd.read_csv(data_path)


def movielens_20m(target='ratings'):
    data_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/movielens_20m/{}.csv'.format(target)
    return pd.read_csv(data_path)
