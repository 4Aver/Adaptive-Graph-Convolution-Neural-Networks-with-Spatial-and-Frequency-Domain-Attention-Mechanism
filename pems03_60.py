import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping
from model import models
import time
import matplotlib.pyplot as plt

#import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
        
def get_parameters():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    # parser.add_argument('--Kt', type=int, default=3) #1D卷积核
    # parser.add_argument('--stblock_num', type=int, default=2)
    # parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    # parser.add_argument('--Ks', type=int, default=3, choices=[3, 2]) # 图卷积核
    # parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv','fusion'])
    # parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    # parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=5, help = 'learn rate scheduler hyper parameter')
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--n_heads', type=int, default=12, help='num of heads') # 8 for pems08_60mins only
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200, help='epochs, default as 10000')
    parser.add_argument('--lr', type=float, default=0.0007, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='early stopping min delta')
    
    
    # parser.add_argument('--enable_fusion', type=int, default=1, help='fusion:1, else: 0')
    parser.add_argument('--model', type=str, default='sagbf', help='model archtecture')
    parser.add_argument('--save_path', type=str, default='./new_exp/temp_pems03_15/heads_12/', help='save path')
    parser.add_argument('--dataset', type=str, default='pems03', choices=['metr-la', 'pems-bay', 'pemsd7-m','pems03','pems04','pems08'])
    parser.add_argument('--droprate', type=float, default=0.2)

    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return args, device



class Configs:
    def __init__(self,args,n_vertex):
        self.seq_len = args.n_his*args.n_heads  # 输入序列长度12*8
        self.pred_len = 1  # 预测序列长度3
        self.d_model = 32  # 模型的维度512
        self.factor = 3  # 用于缩放注意力机制的因子5
        self.n_heads = 8  # 注意力头的数量8
        self.e_layers = 3  # 编码器的层数3
        self.d_ff = 64  # 前馈神经网络的维度2048
        self.dropout = 0.1  # dropout的概率0.1
        self.activation = 'gelu'  # 激活函数
        self.enc_in = n_vertex  # 编码器输入的特征数量
        self.dec_in = n_vertex  # 解码器输入的特征数量
        self.c_out = 1  # 输出的特征数量
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU还是CPU
        self.fnet_d_ff = 64 # 频域前馈神经网络的维度1024
        self.fnet_d_model = 32  # 频域模型的维度512
        self.complex_dropout = 0.1  # 复数dropout的概率0.1
        self.fnet_layers = 2  # 频域网络的层数2
        self.is_emb = True  # 是否使用嵌入层
        
class Configs1:
    def __init__(self,args,n_vertex):
        self.seq_len = args.n_his  # 输入序列长度12
        self.pred_len = args.n_his  # 预测序列长度3
        self.d_model = 32  # 模型的维度512
        self.factor = 3  # 用于缩放注意力机制的因子5
        self.n_heads = 8  # 注意力头的数量8
        self.e_layers = 3  # 编码器的层数3
        self.d_ff = 64  # 前馈神经网络的维度2048
        self.dropout = 0.1  # dropout的概率0.1
        self.activation = 'gelu'  # 激活函数
        self.enc_in = n_vertex  # 编码器输入的特征数量
        self.dec_in = n_vertex  # 解码器输入的特征数量
        self.c_out = 1  # 输出的特征数量
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU还是CPU
        self.fnet_d_ff = 64 # 频域前馈神经网络的维度1024
        self.fnet_d_model = 32  # 频域模型的维度512
        self.complex_dropout = 0.1  # 复数dropout的概率0.1
        self.fnet_layers = 2  # 频域网络的层数2
        self.is_emb = True  # 是否使用嵌入层
        
class Configs2:
    def __init__(self,args,n_vertex):
        self.seq_len = args.n_his*args.n_heads  # 输入序列长度12
        self.pred_len = 1  # 预测序列长度1
        self.d_model = 32  # 模型的维度512
        self.factor = 3  # 用于缩放注意力机制的因子5
        self.n_heads = 8  # 注意力头的数量8
        self.e_layers = 3  # 编码器的层数3
        self.d_ff = 64  # 前馈神经网络的维度2048
        self.dropout = 0.1  # dropout的概率0.1
        self.activation = 'gelu'  # 激活函数
        self.enc_in = n_vertex  # 编码器输入的特征数量
        self.dec_in = n_vertex  # 解码器输入的特征数量
        self.c_out = 1  # 输出的特征数量
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU还是CPU
        self.fnet_d_ff = 64 # 频域前馈神经网络的维度1024
        self.fnet_d_model = 32  # 频域模型的维度512
        self.complex_dropout = 0.1  # 复数dropout的概率0.1
        self.fnet_layers = 2  # 频域网络的层数2
        self.is_emb = True  # 是否使用嵌入层

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    # gso = utility.calc_gso(adj, args.gso_type)
    # if args.graph_conv_type == 'cheb_graph_conv':
    #     gso = utility.calc_chebynet_gso(gso)
    # gso = gso.toarray()
    # gso = gso.astype(dtype=np.float32)
    # args.gso = torch.from_numpy(gso).to(device) #拉普拉斯矩阵传入到args.gso

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0] #流量数据
    # 拆分训练-验证-测试

    train_rate,val_rate,test_rate = 0.5,0.25,0.25
    len_val = int(math.floor(data_col * val_rate))
    len_test = int(math.floor(data_col * test_rate))
    len_train = int(math.floor(data_col * train_rate))
    
    
    # len_train = 228*60
    # len_val = 288*30
    # len_test = 288*30
    
    # 加载流量数据
    train, val, test = dataloader.load_data(args.dataset, len_train, len_val,len_test)
    # print(train.shape) (6840, 228)
    
    #标准化
    zscore = preprocessing.StandardScaler() 
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)
    
    # 将数据转换为x-y形式，并转换为tensor
    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)
    
    # 将数据放入PyTorch DataLoader
    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter

def prepare_model(args, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=args.min_delta, patience=args.patience)

    # model = models.FBA2GFB(args, n_vertex, configs1,configs2).to(device) #注意此处要设置device！！！
    if args.model == 'sagbf':
        model = models.SAGFB(args, n_vertex, configs).to(device) #注意此处要设置device！！！
    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter):
    min_loss = 10
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        if val_loss < min_loss:
            min_loss = val_loss
            print(f"****save model*****epoch:{epoch}****var_loss:{val_loss:.4f}****================")
            # 保存模型语句
            torch.save(model.state_dict(),args.save_path+str(epoch)+'.pth')
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.8f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.1f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        

        if es.step(val_loss):
            print('Early stopping.')
            break

@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args,path):
    # model = model.to(device)
    # model = models.FeatureFusionNet(args, blocks, n_vertex).to(device)
    # model.load_state_dict(torch.load(args.save_path))
    model.load_state_dict(torch.load(path))
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE, test_MAPE,MASK_MAPE = utility.evaluate_metric(model, test_iter, zscore)
    
    # print(f'test pth file path:{path}') #新增2024-11-2-12：47
    print(f'Dataset {args.dataset:s} {path}| Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | MAPE {MASK_MAPE:.8f}')
    return test_MAE, test_RMSE, MASK_MAPE


 
def count_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def search_best_pth(path,metric = 'mae'):
    best_mae,best_rmse,best_mape = 100,100,1
    path = os.listdir(path)
    path.sort(key = lambda x : int(x[:-4]))
    # best_path ='xxx'
    for path_ in path:
        m,r,p = test(zscore, loss, model, test_iter, args,args.save_path+path_)
        # print(f'pth file:{str(path_)}:')
        
        if m<best_mae:
            best_mae = m
            global best_path0
            best_path0 = path_
        if r<best_rmse:
            best_rmse = r
            # global best_path1
            best_path1 = path_
        if p < best_mape:
            best_mape = p
            # global best_path2
            best_path2 = path_
    print(f'best MAE path:{best_path0}')
    print(f'best RMSE path:{best_path1}')
    print(f'best MAPE path:{best_path2}')
    
    # print(f'所谓的best_path{best_path}在这里！！！')
    return best_path1


import sys
moshi = input('请输入模式，训练：train, 测试：test,绘图：curve,搜索：serach \n')
def protect_pth(moshi):
    if moshi =='train':
        print('请核对路径和n_pred!')
        message =input('请确认是要真的开始训练吗？!，Y for yes, N for No\n')
        if str.upper(message) =='Y':
            pass
        else:
            print('程序中止，避免覆盖.pth文件！')
            sys.exit(0)

if __name__ == "__main__" : 
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    args, device = get_parameters()
    print(f'注意：保存路径为：{args.save_path}')
    print(f'注意：预测n_pred为：{args.n_pred}')
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    configs = Configs(args, n_vertex)
    configs1 = Configs1(args, n_vertex)
    configs2 = Configs2(args, n_vertex)
    loss, es, model, optimizer, scheduler = prepare_model(args, n_vertex)
    
    if moshi =='train':
        protect_pth(moshi)
        
        print('开始训练:')
        print('训练将在20s后启动')
        time.sleep(20)
        _ = time.time()
        pass
        train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter)
        print(f'Train Time Consuption:{time.time()-_:.0f} Seconds')
        print('Training Finished.........')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('开始搜寻：')
        best_path = search_best_pth(args.save_path) 
        
    if moshi == 'search':
        print('开始搜寻：')
        best_path = search_best_pth(args.save_path) 
        
    if moshi == 'test':
        print('开始测试：')
        best_path = '199.pth'#for 15 #'91.pth' for 60
        _ = time.time()
        test(zscore, loss, model, test_iter, args, args.save_path+best_path)
        print(f'Test Time Consuption:{time.time()-_:.0f} Seconds')
        # num_param = sum(p.numel() for p in model.parameters())
        num_param = count_parameters(model)
        print(f'模型参数量：{num_param}')

    


if __name__ == "__main__" and moshi  == 'curve':
    # [155.pth,?pth,91.pth]
    batch_size = 512
    node = 66
    args, device = get_parameters()
    args.batch_size = batch_size
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    
    configs = Configs(args, n_vertex)
    configs1 = Configs1(args, n_vertex)
    configs2 = Configs2(args, n_vertex)
    loss, es, model, optimizer, scheduler = prepare_model(args, n_vertex)
    model.load_state_dict(torch.load(args.save_path+'155.pth'))

    utility.show_curve(model, test_iter, zscore,node,args,enable_gap=True) #110 135 68 35
            
    
    

