"""
    Module including the implementation of matrix factorization for recommender system
    Date: 28/Mar/2019
    Author: Li Tang
"""
import numpy as np
import random


class FunkSVD:
    def __init__(self, mat, k, penalty='ridge', penalty_weight=0.5, learning_rate=0.01, learning_rate_decay=0.75,
                 min_learning_rate=None, early_stopping=10):
        self.mat = mat
        self.k = k
        self.penalty = penalty
        self.penalty_weight = penalty_weight
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.mat_p = np.random.rand(len(self.mat), self.k)
        self.mat_q = np.random.rand(self.k, len(self.mat[0]))

    def decom(self, samp_rate=1.0, epochs=20):
        self.learning_rate /= self.learning_rate_decay
        if self.min_learning_rate:
            if self.learning_rate < self.min_learning_rate:
                self.learning_rate = self.min_learning_rate

        for epoch in range(epochs):
            loss = 0
            trained_samples = 0
            skipped_samples = 0
            self.learning_rate *= self.learning_rate_decay
            for row in range(len(self.mat)):
                for col in range(len(self.mat[row])):
                    if np.isnan(self.mat[row, col]):
                        continue
                    if random.random() <= samp_rate:
                        y_hat = np.matmul(self.mat_p[row, :], self.mat_q.T[col, :])

                        if self.penalty == 'ridge':
                            self.mat_p[row, :] = self.mat_p[row, :] + self.learning_rate * (
                                    (self.mat[row, col] - y_hat) * self.mat_q[:, col] -
                                    self.penalty_weight * self.mat_p[row, :])

                            self.mat_q[:, col] = self.mat_q[:, col] + self.learning_rate * (
                                    (self.mat[row, col] - y_hat) * self.mat_p[row, :] -
                                    self.penalty_weight * self.mat_q[:, col])

                            loss += ((self.mat[row, col] - y_hat) ** 2 + self.penalty_weight * (
                                    np.linalg.norm(self.mat_p[row, :]) + np.linalg.norm(self.mat_q.T[col, :]))) / self.k
                        elif self.penalty == 'lasso':
                            pass
                        else:
                            raise ValueError
                        trained_samples += 1
                    else:
                        skipped_samples += 1

            print('epoch:', epoch + 1, '==> loss:', loss)
            print('trained %d samples and skipped %d samples' % (trained_samples, skipped_samples))

    def reco(self, topk=20):
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
