# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:29:10 2024

@author: Xue
version 1 2024-10-22
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import math

# class NodeEmb(nn.Module):
#     # 缺少激活层和全连接层
#     def __init__(self,in_features, embed_dim):
#         super(NodeEmb, self).__init__()
#         self.embed_dim = embed_dim
#         self.in_features = in_features
#         self.f1 = torch.nn.Linear(in_features=self.in_features, out_features=self.embed_dim)
#         self.leaky_relu = nn.LeakyReLU()
#         self.f2 = torch.nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)

#     def forward(self, x):
#         emb = self.f2(self.leaky_relu(self.f1(x)))
#         return emb
# if __name__ == '__main__':
#     x = torch.randn(13,207,12) #batch-nodes-time_in
#     # emb = torch.randn(207,64) #nodes - emb_dim
#     net = NodeEmb(in_features=12, embed_dim=64)
#     #             #time_in   #time_out      #k      #emb_dim
#     print(net(x).shape)

    
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        # print(f'x_g shape:{x_g.shape}') #([13, 3, 207, 12])
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # print(f'x_g shape:{x_g.shape}') #bnki ([13, 207, 3, 12])
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        # print(f'x_gconv shape:{x_gconv.shape}')
        # return node_embeddings,supports,weights,bias
        return x_gconv
if __name__ == '__main__':
    x = torch.randn(13,207,12) #batch-nodes-time_in
    emb = torch.randn(207,64) #nodes - emb_dim
    net = AVWGCN(dim_in=12, dim_out=23, cheb_k=3, embed_dim=64)
    #             #time_in   #time_out      #k      #emb_dim
    print(f'AVWGCN test:{net(x,emb).shape}')

class AdpGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim,num_node):
        super(AdpGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.node_embeddings = nn.Parameter(torch.FloatTensor(num_node, embed_dim), requires_grad=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weights_pool, a=math.sqrt(5))
        init.kaiming_uniform_(self.bias_pool, a=math.sqrt(5))
        init.kaiming_uniform_(self.node_embeddings, a=math.sqrt(5))
        
    def forward(self, x):
        x = x.permute(0,2,1)
        node_embeddings = self.node_embeddings
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        # x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        # print(f'x_g shape:{x_g.shape}') #([13, 3, 207, 12])
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # print(f'x_g shape:{x_g.shape}') #bnki ([13, 207, 3, 12])
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        # print(f'x_gconv shape:{x_gconv.shape}')
        # return node_embeddings,supports,weights,bias
        return x_gconv.permute(0,2,1)     # BTS    
        
if __name__ == '__main__':
    x = torch.randn(13,12,207) #batch-nodes-time_in
    net = AdpGCN(dim_in=12, dim_out=12, cheb_k=3, embed_dim=64,num_node=207)
    #             #time_in   #time_out      #k      #emb_dim
    print(f'AdpGCN test: {net(x).shape}')
    
# AVWGCN test:torch.Size([13, 207, 23])
# AdpGCN test: torch.Size([13, 207, 23])

# x torch.size([13,12,207]) bmc
# torch.Size([207, 64]) node_embeddings
# torch.Size([3, 207, 207]) supports knm
# torch.Size([207, 3, 207, 207]) weights niko
# torch.Size([207, 207]) bias