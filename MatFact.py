"""
    Package including the implementation of matrix factorization for recommender system
    Date: 28/Mar/2019
    Author: Li Tang
"""
import numpy as np


class ListNode:
    def __init__(self, name=None, val=None, prev_node=None, next_node=None):
        self.name = str(name)
        self.val = val
        self.prev_node = prev_node
        self.next_node = next_node
        self.__length = 1

    def set_next(self, next_node):
        self.next_node = next_node
        next_node.prev_node = self

    def set_prev(self, prev_node):
        self.prev_node = prev_node
        prev_node.next_node = self

    # def get_length(self):
    #     dummy = self.next_node
    #     try:
    #         fast_dummy = self.next_node.next_node
    #     except Exception:
    #         self.__length = 2
    #         return self.__length
    #     else:
    #         self.__length = 1
    #         while fast_dummy:
    #             dummy = dummy.next_node
    #             try:
    #                 fast_dummy = fast_dummy.next_node.next_node
    #             except Exception:
    #                 pass
    #             self.__length += 1
    #         return self.__length


class FunkSVD:
    def __init__(self, mat, k, penalty='RIDGE', penalty_weight=0.5, learning_rate=0.01, epochs=5):
        self.mat = mat
        self.k = k
        self.penalty = penalty
        self.penalty_weight = penalty_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mat_p = np.random.rand(len(self.mat), self.k)
        self.mat_q = np.random.rand(self.k, len(self.mat[0]))

    def decom(self):
        for epoch in range(self.epochs):
            loss = 0
            for row in range(len(self.mat)):
                for col in range(len(self.mat[row])):
                    if np.isnan(self.mat[row, col]):
                        continue

                    y_hat = np.matmul(self.mat_p[row, :], self.mat_q.T[col, :])

                    if self.penalty == 'RIDGE':
                        self.mat_p[row, :] = self.mat_p[row, :] + self.learning_rate * (
                                (self.mat[row, col] - y_hat) * self.mat_q[:, col] -
                                self.penalty_weight * self.mat_p[row, :])

                        self.mat_q[:, col] = self.mat_q[:, col] + self.learning_rate * (
                                (self.mat[row, col] - y_hat) * self.mat_p[row, :] -
                                self.penalty_weight * self.mat_q[:, col])

                        loss += (self.mat[row, col] - y_hat) ** 2 + self.penalty_weight * (
                                np.linalg.norm(self.mat_p[row, :]) + np.linalg.norm(self.mat_q.T[col, :]))
                    else:
                        raise ValueError

            print('epoch:', epoch + 1, '==> loss:', loss)

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


mat = np.random.rand(5, 4)
mat[3, 3] = None
mat[3, 2] = None
mat[2, 0] = None
mat[2, 3] = None
mat[0, 2] = None
k = 3
print('mat:', mat)
print('k:', k)
svd = FunkSVD(mat, k, epochs=100)
svd.decom()
result = svd.reco(topk=1)
for i in result:
    print(i)
    print(i, '----->', [j for j in result[i]])
# a = ListNode(name='a', val=1)
# b = ListNode(name='b', val=2)
# c = a
# d = a
# a.set_next(b)
# print(a.name, a.val)
# print(a.next_node.name, a.next_node.val)
# print(b.prev_node.name, b.prev_node.val)
# c = c.next_node
# c.name = 'c'
# c.set_next(d)
# print(c.name, c.val)
# print(a.get_length())
