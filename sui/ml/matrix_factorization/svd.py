"""Module including some implementations of matrix factorization
Date: 28/Mar/2019
Author: Li Tang
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import pickle
import time
from sui.toolbox import top_k
import numpy as np

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.1.7'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class SuiMlMfError(Exception):
    pass


class SVDModel:
    def __init__(self, matrix, k=1, matrix_p=None, matrix_q=None, name='SVD',
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

        self._matrix = None
        self._matrix_min = None
        self._matrix_max = None
        self.set_matrix(matrix)
        self.k = k
        self.matrix_p = np.random.rand(len(self._matrix), self.k) if matrix_p is None else matrix_p
        self.matrix_q = np.random.rand(self.k, len(self._matrix[0])) if matrix_q is None else matrix_q
        self.name = name
        self.version = version

    def set_matrix(self, matrix):
        matrix = np.array(matrix, dtype=np.float64)
        self._matrix_min = np.nanmin(matrix)
        self._matrix_max = np.nanmax(matrix)
        self._matrix = (matrix - self._matrix_min) / (self._matrix_max - self._matrix_min)

    def get_matrix(self):
        print(self._matrix)
        return self._matrix * (self._matrix_max - self._matrix_min) + self._matrix_min

    def train(self):
        raise NotImplementedError

    def predict(self, topk=1, target='p', result_path=None, only_vacancy=True):
        """

        :param topk:
        :param target:
        :param result_path:
        :param only_vacancy:
        :return:
        """
        result_dict = {}
        if target == 'p':
            for row in range(len(self._matrix)):
                topk_reco = []
                for col in range(len(self._matrix[row])):
                    if self._matrix[row, col] is None or np.isnan(self._matrix[row, col]) or only_vacancy is False:
                        score = np.matmul(self.matrix_p[row, :], self.matrix_q.T[col, :])
                        topk_reco.append([col, score])
                result_dict[row] = top_k(data=topk_reco, k=topk, axis=1, desc=True)
        elif target == 'q':
            for col in range(len(self._matrix[0])):
                topk_reco = []
                for row in range(len(self._matrix)):
                    if self._matrix[row, col] is None or np.isnan(self._matrix[row, col]):
                        score = np.matmul(self.matrix_p[row, :], self.matrix_q.T[col, :])
                        topk_reco.append([row, score])
                result_dict[col] = top_k(data=topk_reco, k=topk, axis=1, desc=True)
        else:
            raise SuiMlMfError("'target' should be 'p' or 'q'. Obtained:", target)

        if result_path is not None:
            result_file = '{}_{}.json'.format(self.name, self.version)
            try:
                with open(result_path + result_file, 'w') as result_output:
                    json.dump([result_dict], result_output)
                print('Result is saved into {}.'.format(result_path + result_file))
            except Exception as e:
                raise SuiMlMfError('Failed to save result:', e)

        return result_dict

    def dump(self, model_file_path=None):
        """

        :param model_file_path: Absolute path to dump this model via using pickle.
        :return:
        """
        if model_file_path is None:
            model_file_path = './{}_{}.pkl'.format(self.name, self.version)
        try:
            with open(model_file_path, 'wb') as model_output:
                model_info = dict()
                model_info['matrix'] = self.get_matrix()
                model_info['name'] = self.name
                model_info['version'] = self.version
                model_info['k'] = self.k
                model_info['matrix_p'] = self.matrix_p
                model_info['matrix_q'] = self.matrix_q
                pickle.dump(model_info, model_output)

            print('Model is saved into {}.'.format(model_file_path))

        except Exception as e:
            raise SuiMlMfError('Failed to dump model:', e)

    @staticmethod
    def restore(model_file_path):
        raise NotImplementedError
