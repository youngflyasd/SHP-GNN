import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, degree, combine):
        super(SGC, self).__init__()
        self.dropout = 0.5
        self.degree = degree
        self.combine = combine

        self.W = nn.Linear(nfeat, nclass)
        self.W2 = nn.Linear(nfeat*(self.degree+1), nclass)
       
        self.D = nn.Dropout(self.dropout)
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)       #在特征分类前，添加一层dropout
        r_x = x
        if self.combine == 'cat':           
            output = self.W2(x)
        elif self.combine == 'max':
            output = self.W(x)
        elif self.combine == 'ssgc':
            output = self.W(x)

        return output, r_x

class GraphConvolution(Module):
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
        output = torch.spmm(adj, support)

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

def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0, cuda=True, degree = 16, combine = 'cat'):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat, nclass=nclass, degree = degree, combine = combine)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model
