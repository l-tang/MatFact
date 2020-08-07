"""
Date:
Author:
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import random
import sys
import time
import numpy as np
from .svd import SVDModel

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.1.7'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class SuiSVDppError(Exception):
    pass


class SVDpp(SVDModel):
    def __init__(self, matrix, k=1, matrix_p=None, matrix_q=None, name='SVDpp',
                 version=time.strftime("%Y%m%d", time.localtime())):
        """

        :param matrix:
        :param k:
        :param matrix_p:
        :param matrix_q:
        :param name:
        :param version:
        """
        assert matrix is not None, "'matrix' cannot be None."
        assert k > 0 and isinstance(k, int), "'k' should be an integer greater than 0."

        super().__init__(matrix=matrix, k=k, matrix_p=matrix_p, matrix_q=matrix_q, name=name, version=version)

    # TODO
    def train(self):
        pass

    # TODO
    def __fit(self):
        pass

    # TODO
    @staticmethod
    def restore(model_file_path):
        pass
