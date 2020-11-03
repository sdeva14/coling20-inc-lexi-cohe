import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.parameter import Parameter
from torch.nn import Module

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.elu = nn.ELU()
        self.leak_relu = nn.LeakyReLU()

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            # x = F.relu(x)
            x = self.leak_relu(x)
            # x = self.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.gc2(x, adj)
        
        # if use_relu:
        #     x = self.elu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc3(x, adj)

        return x

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init_param()

    def init_param(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        # output = torch.spmm(adj, support)  # cannot handle batch, but more efficient
        # output = torch.matmul(adj.to_dense(), support)  # to handle batch-processing

        output = torch.matmul(adj, support)  # assume already densed repr

        return output

# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output