import argparse
import time
import trainonclean
import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn import*

import NSatk_GMA


#1.GCN在普通数据训练
# Load data注意我这里做了一个adj_org，这个代表的是没有做归一化的adj
adj,adj_org,features, labels, idx_train, idx_val, idx_test = load_data(path='./data_cora/cora/')

cleanGCN = trainonclean.cleanGCN(adj, features, labels, idx_train, idx_val, idx_test)

# Train model
t_total = time.time()
for epoch in range(200):
    cleanGCN.train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

#提取W1，W2
W1 = cleanGCN.W1
W2 = cleanGCN.W2

#2.毒害数据
#target
u = [0,1]
#这里adj先传普通的，在内部再计算D^-1
NSatk = NSatk_GMA(adj, features, labels, W1, W2, u)


#3.GCN在有毒的数据上训练

#4.GCN在原数据上预测

#5.GCN在有毒数据上预测