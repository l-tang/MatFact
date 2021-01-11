"""Module including some implementations of matrix factorization
Date: 28/Mar/2019
Author: Li Tang
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


class SuiFunkSVDError(Exception):
    pass


class FunkSVD(SVDModel):
    def __init__(self, matrix, k=1, matrix_p=None, matrix_q=None, name='FunkSVD',
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

    def train(self, penalty='ridge', penalty_weight=0.5, learning_rate=0.75, learning_rate_decay=1.0,
              min_learning_rate=None, dropout=0.0, epochs=50, early_stopping=10, workers=1):
        """
        :param penalty: penalty can be either lasso or ridge; hybrid of them is not allowed
        :param penalty_weight: weight of the given type of penalty
        :param learning_rate: initial learning rate
        :param learning_rate_decay: decay
        :param min_learning_rate:
        :param dropout:
        :param epochs:
        :param early_stopping:
        :param workers:
        :return:
        """
        assert penalty in ['ridge', 'lasso'], "'penalty' should be either 'ridge' or 'lasso'."
        assert penalty_weight > 0, "'penalty_weight' should be greater than 0."
        assert learning_rate > 0, "'learning_rate' should be greater than 0."
        assert learning_rate_decay > 0, "'learning_rate_decay' should be greater than 0." \
                                        " Set 1 for no decay in training."
        assert 0 <= dropout < 1.0, "The domain of definition of 'dropout' should be [0, 1)."
        assert isinstance(early_stopping, int), \
            "'early_stopping' should be an integer. Set 0 or any negative integer to interdict early stopping."
        assert isinstance(workers, int) and workers > 0, "'workers' should be an integer greater than 0."

        learning_rate /= 1 - dropout

        loss_history = [sys.maxsize]
        for epoch in range(epochs):
            loss = 0
            trained_samples = 0
            skipped_samples = 0

            if min_learning_rate:
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

            # TODO: split matrix into multiple sections for concurrency
            row_purview_list = [(0, len(self._matrix))]
            col_purview_list = [(0, len(self._matrix[0]))]

            # start to train
            for section_idx in range(len(row_purview_list)):
                section_loss, section_trained_samples, section_skipped_samples = self.__fit(
                    row_purview_list[section_idx], col_purview_list[section_idx], learning_rate, penalty,
                    penalty_weight, dropout)
                loss += section_loss
                trained_samples += section_trained_samples
                skipped_samples += section_skipped_samples

            print('epoch: {} ==> loss: {}'.format(epoch + 1, loss))

            if dropout > 0:
                print('Trained {} samples and skipped {} samples.'
                      ' The dropout rate is: {}%'.format(trained_samples, skipped_samples, round(
                    skipped_samples / (trained_samples + skipped_samples) * 100)))

            if learning_rate_decay != 1.0:
                print('Current learning rate: {}'.format(learning_rate))
                learning_rate *= learning_rate_decay

            if early_stopping > 0:
                if loss < loss_history[0]:
                    loss_history = [loss]
                else:
                    loss_history.append(loss)

                if len(loss_history) >= early_stopping:
                    print(
                        'Early stopping! The best performance is at No.{} epoch and the loss have not been decreased'
                        ' from then on as {}:'.format(epoch - early_stopping + 2, loss_history))
                    break
            else:
                continue

    def __fit(self, row_purview: tuple, col_purview: tuple, learning_rate, penalty, penalty_weight, dropout):
        """

        :param row_purview:
        :param col_purview:
        :param learning_rate:
        :param penalty:
        :param penalty_weight:
        :param dropout:
        :return:
        """
        loss = 0
        trained_samples = 0
        skipped_samples = 0

        for row in range(row_purview[0], row_purview[1]):
            for col in range(col_purview[0], col_purview[1]):
                if self._matrix[row, col] is None or np.isnan(self._matrix[row, col]):
                    continue
                if random.random() <= 1 - dropout:
                    y_hat = np.matmul(self.matrix_p[row, :], self.matrix_q.T[col, :])

                    if penalty == 'ridge':
                        self.matrix_p[row, :] += learning_rate * ((self._matrix[row, col] - y_hat) *
                                                                  self.matrix_q[:, col] - penalty_weight *
                                                                  self.matrix_p[row, :]) / self.k

                        self.matrix_q[:, col] += learning_rate * ((self._matrix[row, col] - y_hat) *
                                                                  self.matrix_p[row, :] - penalty_weight *
                                                                  self.matrix_q[:, col]) / self.k

                        loss += ((self._matrix[row, col] - y_hat) ** 2 + penalty_weight * (
                                np.linalg.norm(self.matrix_p[row, :]) + np.linalg.norm(
                            self.matrix_q.T[col, :]))) / self.k
                    elif penalty == 'lasso':
                        self.matrix_p[row, :] += learning_rate * ((self._matrix[row, col] - y_hat) *
                                                                  self.matrix_q[:, col] - penalty_weight) / self.k
                        self.matrix_q[:, col] += learning_rate * ((self._matrix[row, col] - y_hat) *
                                                                  self.matrix_p[row, :] - penalty_weight) / self.k
                        loss += ((self._matrix[row, col] - y_hat) ** 2 + penalty_weight * (
                                np.linalg.norm(self.matrix_p[row, :], ord=1) + np.linalg.norm(
                            self.matrix_q.T[col, :], ord=1))) / self.k
                    else:
                        raise ValueError
                    trained_samples += 1
                else:
                    skipped_samples += 1

        return loss, trained_samples, skipped_samples

    @staticmethod
    def restore(model_file_path):
        """

        :param model_file_path:
        :return:
        """
        try:
            with open(model_file_path, 'rb') as model_input:
                model_info = pickle.load(model_input)
                model = FunkSVD(**model_info)
            print('Model is loaded from {}.'.format(model_file_path))
            return model

        except Exception as e:
            raise SuiFunkSVDError('Failed to restore model:', e)

    # TODO: add new user or product into a pretrained matrix
    def add(self, target, value, initializer='mean'):
        """

        :param target:
        :param value:
        :param initializer:
        :return:
        """
        assert target in ['k', 'matrix_p', 'matrix_q'], "'target' cannot be found."
        assert isinstance(value, int) and value > 0, "'value' should be an integer greater than 0."
        assert initializer in ['mean', 'random'], "'initializer' should be either 'mean' or 'random'."
