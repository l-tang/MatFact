"""
    Module including the implementation of matrix factorization for recommender system
    Date: 28/Mar/2019
    Author: Li Tang
"""
import sys
import numpy as np
import random


class FunkSVD:
    def __init__(self, mat, k: int, penalty='ridge', penalty_weight=0.5, learning_rate=0.05, learning_rate_decay=0.85,
                 min_learning_rate=None):
        """

        :param mat:
        :param k:
        :param penalty:
        :param penalty_weight:
        :param learning_rate:
        :param learning_rate_decay:
        :param min_learning_rate:
        """
        assert mat is not None
        assert k > 0 and isinstance(k, int)
        assert penalty in ['ridge', 'lasso']
        assert penalty_weight > 0
        assert learning_rate > 0
        assert learning_rate_decay > 0

        self.mat = mat
        self.k = k
        self.penalty = penalty
        self.penalty_weight = penalty_weight
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.mat_p = np.random.rand(len(self.mat), self.k)
        self.mat_q = np.random.rand(self.k, len(self.mat[0]))

    def decom(self, dropout=0.0, epochs=20, early_stopping=10):
        """

        :param dropout:
        :param epochs:
        :param early_stopping:
        :return:
        """
        assert 0 <= dropout < 1.0
        assert isinstance(early_stopping, int)

        self.learning_rate /= self.learning_rate_decay / (1 - dropout)
        if self.min_learning_rate:
            if self.learning_rate < self.min_learning_rate:
                self.learning_rate = self.min_learning_rate

        loss_history = [sys.maxsize]
        for epoch in range(epochs):
            loss = 0
            trained_samples = 0
            skipped_samples = 0
            self.learning_rate *= self.learning_rate_decay
            for row in range(len(self.mat)):
                for col in range(len(self.mat[row])):
                    if not self.mat[row, col] or np.isnan(self.mat[row, col]):
                        continue
                    if random.random() <= 1 - dropout:
                        y_hat = np.matmul(self.mat_p[row, :], self.mat_q.T[col, :])

                        if self.penalty == 'ridge':
                            self.mat_p[row, :] += self.learning_rate * (
                                    (self.mat[row, col] - y_hat) * self.mat_q[:, col] -
                                    self.penalty_weight * self.mat_p[row, :])

                            self.mat_q[:, col] += self.learning_rate * (
                                    (self.mat[row, col] - y_hat) * self.mat_p[row, :] -
                                    self.penalty_weight * self.mat_q[:, col])

                            loss += ((self.mat[row, col] - y_hat) ** 2 + self.penalty_weight * (
                                    np.linalg.norm(self.mat_p[row, :]) + np.linalg.norm(self.mat_q.T[col, :]))) / self.k
                        elif self.penalty == 'lasso':
                            self.mat_p[row, :] += self.learning_rate * (
                                        (self.mat[row, col] - y_hat) * self.mat_q[:, col] - self.penalty_weight)
                            self.mat_q[:, col] += self.learning_rate * (
                                        (self.mat[row, col] - y_hat) * self.mat_p[row, :] - self.penalty_weight)
                            loss += ((self.mat[row, col] - y_hat) ** 2 + self.penalty_weight * (
                                        np.linalg.norm(self.mat_p[row, :], ord=1) + np.linalg.norm(self.mat_q.T[col, :],
                                                                                                   ord=1))) / self.k
                        else:
                            raise ValueError
                        trained_samples += 1
                    else:
                        skipped_samples += 1

            print('epoch: {} ==> loss: {}'.format(epoch + 1, loss))
            print('trained {} samples and skipped {} samples'.format(trained_samples, skipped_samples))

            if early_stopping > 0:
                if loss < loss_history[0]:
                    loss_history = [loss]
                else:
                    loss_history.append(loss)

                if len(loss_history) >= early_stopping:
                    print(
                        'Early stopping! The best performance is at No.{} epoch and the loss have not been decreased from then on as {}:'.format(
                            epoch - early_stopping + 2, loss_history))
                    break
            else:
                continue

    def reco(self, topk=20):
        """

        :param topk:
        :return:
        """
        result_dict = {}
        for row in range(len(self.mat)):
            topk_reco = []
            score_list = []
            for col in range(len(self.mat[row])):
                if np.isnan(self.mat[row, col]):
                    score = np.matmul(self.mat_p[row, :], self.mat_q.T[col, :])
                    if len(topk_reco) < topk:
                        topk_reco.append(col)
                        score_list.append(score)
                    elif min(score_list) < score:
                        idx = score_list.index(min(score_list))
                        topk_reco[idx] = col
                        score_list[idx] = score
                    else:
                        continue
            result_dict[row] = iter(topk_reco)
        return result_dict


mat = np.array(
    [[0, 1, 2, 1, 2], [0, 2, 2, np.nan, np.nan], [np.nan, 0, 1, np.nan, np.nan], [np.nan, None, None, 1, 2],
     [1, 0, 2, None, None]])
funksvd = FunkSVD(mat=mat, k=3)
funksvd1 = FunkSVD(mat=mat, k=3, penalty='lasso')
funksvd.decom(epochs=100, early_stopping=-1)
print(funksvd.mat_p, funksvd.mat_q)
funksvd1.decom(epochs=100, early_stopping=-1)
print(funksvd1.mat_p, funksvd1.mat_q)
