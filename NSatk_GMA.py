import numpy as np
import scipy.sparse as sp

#本算法目前有点问题，1.adj计算时应该使用归一化之后的；2.loss差需要再留意一些；3.计算速度
#4.对于无向图改动，改一条边需要改动两个值； 5.对于GCN预测的结果，需要原值

class nsatk_GMA:
    """

    """

    def __init__(self, adj, features, labels, W1, W2, u):
        self.W1 = W1
        self.W2 = W2
        self.adj = adj.copy()
        self.adj_org = adj.copy()
        self.u = u  # 一个node set
        self.features = features
        self.labels = labels

        self.W = self.W1.dot(self.W2)
        self.N = adj.shape[0]

    def compute_struc_loss(self, edge):
        """
        计算单独某一条边的在整个node set上的loss得分，关于计算loss，nettack使用了简单化的方式，即在计算A时，
        只算入临近的节点，但那个是对单攻击而非对群。对群方面因为需要评估复数个节点的loss，所以为了方便计算，只能为
        复数个节点分别聚合其邻居或者干脆直接使用A原本的邻接矩阵计算。
        :param edge: nparray
        :return: float
        """
        A = self.adj
        A[edge] = 1

        XW = self.compute_XW()
        Z_new = A.dot(XW)
        Z_new = A.dot(Z_new)

        Z_org = self.adj.dot(XW)
        Z_org = self.adj.dot(Z_org)

        """

        Z_new = A.dot(self.compute_XW())
        Z_new = A.dot(Z_new)
        Z_new = Z_new.dot(self.W2)

        Z_org = self.adj.dot(self.compute_XW())
        Z_org = self.adj.dot(Z_org)
        Z_org = Z_org.dot(self.W2)
        """
        loss = Z_new - Z_org  # 新旧预测之差
        loss = loss[self.u]  # node set上新旧预测之差

        Z = self.labels[self.u]
        Z_index = np.where(Z == 1)

        # Z_index = np.column_stack((np.arange(len(Z_index[1])),Z_index[1]))
        loss = loss[Z_index]  # 在正确label上的预测降低程度

        return loss.sum()

    def compute_XW(self):
        """

        :param X_obs:
        :param W:
        :return:
        """
        return self.features.dot(self.W)

    def slect_srtuc_atk(self):
        """
        计算node set与所有节点的struc loss，返回要选择的扰动的边,我认为这里应该一次性选取足够的边。然后再进行扰动联系降低
        :return:
        """
        potential_edges = np.row_stack([np.column_stack((np.tile(node, self.N), np.arange(self.N))) for node in self.u])
        potential_edges_loss = []
        for i in range(len(potential_edges)):
            potential_edges_loss.append(self.compute_struc_loss(potential_edges[i]))
            print(i)

        tgt = np.argsort(potential_edges_loss)[-1]
        print(potential_edges_loss[tgt])
        return potential_edges[tgt]






















