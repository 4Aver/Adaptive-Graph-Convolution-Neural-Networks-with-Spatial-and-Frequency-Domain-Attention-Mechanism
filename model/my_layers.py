# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 07:07:09 2024

@author: Xue
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import einops 

##统一为BCTS顺序 时空

#通道对齐 C_in to C_out,降低或者升高通道数，降维或者升维（用0 padding)
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        
        return x
if __name__ == '__main__':
    
    f = torch.randn(13,1,12,229)
    N = Align(c_in=1, c_out=8)
    print('4D对齐')
    print(N(f).size()) #13-8-12-229   

class Align3d(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align3d, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, 
                                          timestep]).to(x)], dim=1)
        else:
            x = x
        return x
if __name__ == '__main__':
    
    f = torch.randn(13,12,229)
    print('3D对齐')
    N = Align3d(c_in=12, c_out=512)
    print(N(f).size()) #torch.Size([13, 512, 229])  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)   # 从1开始到最后面，补长为2，其实代表的就是奇数位置
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]  经过词向量的输入
        """
        x = x + self.pe[:, :x.size(1),:x.size(2)].clone().detach()   # 经过词向量的输入与位置编码相加
        x = self.dropout(x)
        return x

      
if __name__ == '__main__':
    pos_encoding_s = PositionalEncoding(d_model=12,dropout=0)
    xs = torch.randn(13,228,12) #BST ->BST
    print('位置编码模块')
    print(pos_encoding_s(xs).shape) # torch.Size([13, 228, 12])    

class PositionalEncodingLayer(nn.Module):
    """实现Positional Encoding功能"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.max_len =max_len
        if self.d_model % 2 ==0:
            self.pe_encoder = PositionalEncoding(self.d_model,self.dropout)
        else:
            self.pe_encoder = PositionalEncoding(self.d_model+1,self.dropout)


    def forward(self, x):
        
        """
        x: [seq_len, batch_size, d_model]  经过词向量的输入
        """
        return self.pe_encoder(x)
if __name__ == '__main__':    
    pos_encoding_t = PositionalEncodingLayer(d_model=229,dropout=0)
    xt = torch.randn(13,12,229) # BTS -> BTS
    print('时间轴位置编码layer')
    print(pos_encoding_t(xt).shape) #([13, 12, 229]),词向量+位置编码




'''
tokens = 12
d_model = 229
'''
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,8))
# # plt.pcolormesh(输入=2Dtensor)
# plt.pcolormesh(pos_encoding_t(xt)[0], cmap='viridis')
# plt.xlabel('Embedding Dimensions')
# # plt.xlim((0, d_model))
# # plt.ylim((tokens,0))
# plt.ylabel('Token Position')
# plt.colorbar()
# plt.show()

import einops 
def to_3d(x):
    return einops.rearrange(x, 'b c s t -> b s (t c)')

# (b,h*w,c)->(b,c,h,w)

def to_4d(x,s,n):
    return einops.rearrange(x, 'b t (s n) -> b t n s',s=s,n=n)
if __name__ == '__main__':  
    f_t  = torch.randn(13,12,207*8)
    print('3d to 4d')
    print(to_4d(f_t,s=207,n=8).shape)
    
    f_s  = torch.randn(13,207,12*8)
    print('3d to 4d')
    print(to_4d(f_s,s=12,n=8).shape)

class Mha(nn.Module):
    def __init__(self,in_features, embed_dim, num_heads, droprate):
        super(Mha, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.droprate = droprate
        self.in_features = in_features
        self.fcq1 = torch.nn.Linear(in_features=self.in_features, out_features=self.embed_dim)
        self.fck1 = torch.nn.Linear(in_features=self.in_features, out_features=self.embed_dim)
        self.fcv1 = torch.nn.Linear(in_features=self.in_features, out_features=self.embed_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.fcq2 = torch.nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.fck2 = torch.nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.fcv2 = torch.nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.attnlayer = torch.nn.MultiheadAttention(self.embed_dim, self.num_heads, self.droprate)

    def forward(self, x):
        q = self.fcq2(self.leaky_relu(self.fcq1(x)))
        k = self.fck2(self.leaky_relu(self.fck1(x)))
        v = self.fcv2(self.leaky_relu(self.fcv1(x)))
        # x = torch.add(x,self.attnlayer(q,k,v)[0]) 
        x = self.attnlayer(q,k,v)[0]
        return x
    
if __name__ == '__main__':    
    Nt = Mha(in_features=229,embed_dim =256*8,num_heads=8,droprate=0.3)
    ft = torch.randn(13,12,229)
    print('temporal attention')
    print(Nt(ft).size()) #13-12-256*8
    
    Ns = Mha(in_features=12,embed_dim =64*8,num_heads=8,droprate=0.3)
    fs = torch.randn(13,229,12)
    print('spatial attention')
    print(Ns(fs).shape) #13-229-64*8

class TAT3D(nn.Module):
    # 统一为时间-空间顺序,时间轴 注意力，将时间12作为embedding输入： 13*12*207 ==》13*12*（207*8）
    def __init__(self,temp_dim, spat_dim,num_heads, droprate):
        super(TAT3D, self).__init__()
        self.ft = Mha(in_features=spat_dim, embed_dim=spat_dim*num_heads, num_heads=num_heads, droprate=droprate)
    def forward(self, x):
        # [batch - timp - spat]
        x_t = self.ft(x) # BTS-->BST->BTS
        return x_t
    
if __name__ == '__main__':  
    FF= TAT3D(temp_dim=12, spat_dim=207, num_heads=8, droprate=0.1)
    ft = torch.randn(13,12,207)
    print(FF(ft).shape) #([13, 12, 207*8])
    
    
    
class SAT3D(nn.Module):
    # 统一为时间-空间顺序 空间轴注意力，将空间207作为embedding 输入 13*12*207 =>13 * 207 *12 =>13*207*(12*8)
    def __init__(self,temp_dim, spat_dim,num_heads, droprate):
        super(SAT3D, self).__init__()
        self.fs = Mha(in_features=temp_dim, embed_dim=temp_dim*num_heads, num_heads=num_heads, droprate=droprate)
    def forward(self, x):
        # [batch - timp - spat]
        x_s = self.fs(x.permute(0,2,1)).permute(0,2,1) # BTS
        
        return x_s
    
if __name__ == '__main__':  
    FF= SAT3D(temp_dim=12, spat_dim=207, num_heads=8, droprate=0.1)
    fs = torch.randn(13,12,207)
    print(FF(fs).shape) #([13, 17, 12, 229])

class FeatureFusion4D(nn.Module):
    # BST to BST
    def __init__(self,temp_dim, spat_dim,num_heads, droprate):
        super(FeatureFusion4D, self).__init__()
        self.temp_dim = temp_dim
        self.spat_dim = spat_dim
        self.num_heads = num_heads
        self.droprate = droprate
        self.pe_s = PositionalEncodingLayer(d_model=self.temp_dim,dropout=self.droprate)
        self.pe_t = PositionalEncodingLayer(d_model=self.spat_dim,dropout=self.droprate)
        self.attn_s = Mha(in_features=self.temp_dim,embed_dim=self.temp_dim*self.num_heads,
                          num_heads=self.num_heads,droprate=self.droprate)
        self.attn_t = Mha(in_features=self.spat_dim,embed_dim=self.spat_dim*self.num_heads,
                          num_heads=self.num_heads,droprate=self.droprate)
        self.align_s = Align3d(c_in=self.spat_dim, c_out=self.spat_dim*self.num_heads)
        self.align_t = Align3d(c_in=self.temp_dim, c_out=temp_dim*self.num_heads)
        self.temp_fuse = nn.Conv1d(in_channels=self.spat_dim, out_channels=self.spat_dim, kernel_size=self.num_heads,stride=1,dilation=self.temp_dim)
        self.spat_fuse = nn.Conv1d(in_channels=self.temp_dim, out_channels=self.temp_dim, kernel_size=self.num_heads,stride=1,dilation=self.spat_dim)

    def forward(self, x):
        x_s = self.pe_s(x) # BST - BST
        x_s = self.attn_s(x) # B-S-T*n
        x_s = to_4d(x_s, s=self.temp_dim, n = self.num_heads) #B-S-T*n to B-S-n-T
        x_s = x_s.permute(0,2,3,1) #B-S-n-T to B-n-T-S

        x_t = self.pe_t(x.permute(0,2,1)) # BST to B-T-S
        x_t = self.attn_t(x_t) #B-T-S*n
        x_t = to_4d(x_t, s=self.spat_dim, n=self.num_heads)#B-T-S*n to B-T-n-S
        x_t = x_t.permute(0,2,1,3) #B-T-n-S to B-n-T-S
        x_res = x.unsqueeze(1) # BST to BCST
        x_res = x_res.permute(0,1,3,2) #BCST to BCTS
        
        return torch.concat((x_t,x_s,x_res),dim=1)
        
if __name__ == '__main__':  
    FF= FeatureFusion4D(temp_dim=12, spat_dim=229, num_heads=8, droprate=0.1)
    ft = torch.randn(13,229,12)
    print(FF(ft).shape) #([13, 17, 12, 229])
