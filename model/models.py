import torch
import torch.nn as nn

from model.my_layers import TAT3D, SAT3D, to_4d
from model.F_Block import F_Block
from model.adp_gcn_v1 import AdpGCN
from model.layers import OutputBlock
  
class SAGFB(nn.Module):
    '''
    T: Spat attn
    AG: adaptive graph conv
    FB:F_block : Config
    X: BTS
    '''
    def __init__(self,args,n_vertex,configs):
        super(SAGFB,self).__init__()
        self.s_attn = SAT3D(args.n_his, n_vertex,args.n_heads, args.droprate) #13*12*8 207
        self.adgcn = AdpGCN(dim_in=args.n_his*args.n_heads, dim_out=args.n_his*args.n_heads, 
                            cheb_k=3, embed_dim=args.emb_dim,num_node=n_vertex) # BTS 13 * 12*8 *207
        self.f_block = F_Block(configs) #BTS
        
    def forward(self,x):
        x = x.squeeze(1) #BCTS =>BTS B-12-207
        x = self.s_attn(x) #13-12*8-207
        x = self.adgcn(x)+x # BTS 13-12*8-207
        x = self.f_block(x) #BTS to BTS B-1-207
        x = x.unsqueeze(1) # B-1-1-S B-1-207 to B-1-1-207
        return x
        
class FB(nn.Module):
    '''
    T:temp attn
    AG: adaptive graph conv
    FB:F_block
    X: BTS
    '''
    def __init__(self,args,n_vertex,configs):
        super(FB,self).__init__()
        # self.t_attn = TAT3D(args.n_his, n_vertex,args.n_heads, args.droprate) #13*12* 207*8
        self.adgcn = AdpGCN(dim_in=args.n_his, dim_out=args.n_his, cheb_k=3, embed_dim=args.emb_dim,num_node=n_vertex) # BTS
        self.f_block = F_Block(configs) #BTS
        
    def forward(self,x):
        x = x.squeeze(1) #BCTS =>BTS B-12-207
        # x = self.adgcn(x) # BTS 13-12-207
        x = self.f_block(x) #BTS to BTS B-1-207
        x = x.unsqueeze(1) # B-1-1-S B-1-207 to B-1-1-207
        return x       

    
class SAGFBO(nn.Module):
    def __init__(self,args,n_vertex,configs1):
        super(SAGFBO,self).__init__()
        self.s_attn = SAT3D(args.n_his, n_vertex,args.n_heads, args.droprate) #13*12*8 207
        self.adgcn = AdpGCN(dim_in=args.n_his*args.n_heads, dim_out=args.n_his*args.n_heads, 
                            cheb_k=3, embed_dim=args.emb_dim,num_node=n_vertex) # BTS 13 * 12*8 *207
        self.f_block = F_Block(configs1) #BTS 13 * 12*8 *207
        self.t_len = args.n_his
        self.heads = args.n_heads
        self.outputblock = OutputBlock(args.n_his,args.n_heads,[32,64],1,n_vertex,'glu',True,0.1)
        
    def forward(self,x):
        x = x.squeeze(1) #BCTS =>BTS B-12-207
        x = self.s_attn(x) #13-12*8-207
        x = self.adgcn(x)+x # BTS 13-12*8-207
        x = self.f_block(x) #BTS 13-12*8-207
        x = x.permute(0,2,1) #BST 13-207-12*8
        x = to_4d(x,self.heads,self.t_len) # B-S-T-Heads 13-207-12-8
        x = x.permute(0,3,2,1) # B-Heads(C) - T-S 13-8-12-207
        x = self.outputblock(x)
        
        return x

class AGFB(nn.Module):
    '''
    T: Spat attn
    AG: adaptive graph conv
    FB:F_block : Config
    X: BTS
    '''
    def __init__(self,args,n_vertex,configs):
        super(AGFB,self).__init__()
        # self.s_attn = SAT3D(args.n_his, n_vertex,args.n_heads, args.droprate) #13*12*8 207
        self.adgcn = AdpGCN(dim_in=args.n_his*1, dim_out=args.n_his*1, 
                            cheb_k=3, embed_dim=args.emb_dim,num_node=n_vertex) # BTS 13 * 12*8 *207
        self.f_block = F_Block(configs) #BTS
        
    def forward(self,x):
        x = x.squeeze(1) #BCTS =>BTS B-12-207
        # x = self.s_attn(x) #13-12*8-207
        x = self.adgcn(x)+x # BTS 13-12-207
        x = self.f_block(x) #BTS to BTS B-1-207
        x = x.unsqueeze(1) # B-1-1-S B-1-207 to B-1-1-207
        return x

class SFB(nn.Module):
    '''
    T: Spat attn
    AG: adaptive graph conv
    FB:F_block : Config
    X: BTS
    '''
    def __init__(self,args,n_vertex,configs):
        super(SFB,self).__init__()
        self.s_attn = SAT3D(args.n_his, n_vertex,args.n_heads, args.droprate) #13*12*8 207
        self.adgcn = AdpGCN(dim_in=args.n_his*args.n_heads, dim_out=args.n_his*args.n_heads, 
                            cheb_k=3, embed_dim=args.emb_dim,num_node=n_vertex) # BTS 13 * 12*8 *207
        self.f_block = F_Block(configs) #BTS
        
    def forward(self,x):
        x = x.squeeze(1) #BCTS =>BTS B-12-207
        x = self.s_attn(x) #13-12*8-207
        # x = self.adgcn(x)+x # BTS 13-12-207
        x = self.f_block(x) #BTS to BTS B-1-207
        x = x.unsqueeze(1) # B-1-1-S B-1-207 to B-1-1-207
        return x
    
class SAGO(nn.Module):
    def __init__(self,args,n_vertex,configs1):
        super(SAGO,self).__init__()
        self.s_attn = SAT3D(args.n_his, n_vertex,args.n_heads, args.droprate) #13*12*8 207
        self.adgcn = AdpGCN(dim_in=args.n_his*args.n_heads, dim_out=args.n_his*args.n_heads, 
                            cheb_k=3, embed_dim=args.emb_dim,num_node=n_vertex) # BTS 13 * 12*8 *207
        # self.f_block = F_Block(configs1) #BTS 13 * 12*8 *207
        self.t_len = args.n_his
        self.heads = args.n_heads
        self.outputblock = OutputBlock(args.n_his,args.n_heads,[32,64],1,n_vertex,'glu',True,0.1)
        
    def forward(self,x):
        x = x.squeeze(1) #BCTS =>BTS B-12-207
        x = self.s_attn(x) #13-12*8-207
        x = self.adgcn(x)+x # BTS 13-12*8-207
        # x = self.f_block(x) #BTS 13-12*8-207
        x = x.permute(0,2,1) #BST 13-207-12*8
        x = to_4d(x,self.heads,self.t_len) # B-S-T-Heads 13-207-12-8
        x = x.permute(0,3,2,1) # B-Heads(C) - T-S 13-8-12-207
        x = self.outputblock(x)
        
        return x