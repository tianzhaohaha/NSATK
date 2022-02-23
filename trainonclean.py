# Training settings
import argparse
import time

from torch import optim

from pygcn import *
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
"""
adj,adj_org,features, labels, idx_train, idx_val, idx_test = load_data(path='./data_cora/cora/')
"""
#考虑到之后adj与feature不同，所以类定义时需要重新导入
#带来的变化就是，model在类内定义
class cleanGCN:
    def __init__(self,adj,features, labels, idx_train, idx_val, idx_test):
        # Model and optimizer
        self.model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

        if args.cuda:
            self.model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        self.W1 = self.model.gc1.weight
        self.W2 = self.model.gc2.weight
        self.features = features
        self.adj = adj
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.labels = labels
        self.idx_val = idx_val
    def train(self,epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self.model.eval()
            output = self.model(self.features, self.adj)

        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        self.W1 = self.model.gc1.weight
        self.W2 = self.model.gc2.weight

    def test(self):
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))









