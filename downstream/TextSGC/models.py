import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class SGC(nn.Module):
    def __init__(self, nfeat, nclass, degree, combine, bias=False):
        super(SGC, self).__init__()
        self.dropout = 0.95
        self.degree = degree
        self.combine = combine
        
        self.W = nn.Linear(nfeat, nclass, bias=bias)
        
        self.W2 = nn.Linear(nfeat, nclass, bias=bias)

        torch.nn.init.xavier_normal_(self.W.weight)
        torch.nn.init.xavier_normal_(self.W2.weight)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.combine == 'cat':
            out = self.W2(x)
        elif self.combine == 'max':
            out = self.W(x)
        return out

class GraphConvolution(nn.Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.mm(adj, support)

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x