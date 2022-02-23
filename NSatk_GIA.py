import numpy as np
import utils
import scipy as sp

class nsatk:
    def __init__(self,adj, feature, label, W1, W2, u):
        self.W1 = W1
        self.W2 = W2
        self.adj = adj.copy()
        self.adj_org = adj.copy()
        self.u = u  # 一个node set
        self.feature = feature
        self.label = label

        self.W = self.W1.dot(self.W2)
        self.N = adj.shape[0]


    def compute_A_hat(self,adj):
        """
        构造对称矩阵，也就是又想向图转为无向图。并将矩阵做归一化，也即求A_hat
        :param adj:
        :return:
        """
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = utils.normalize(adj + sp.eye(adj.shape[0]))

        return adj

    def compute_XW(self):
        """
        简化计算，因为选择二层GCN，所以先把X*W1*W2计算出来，之后便好计算A_hat*A_hat*(X*W1*W2)
        :return:
        """
        return self.feature.dot(self.W)

    def injected_node_degree(self):

        num = len(self.u)/2
        return num














