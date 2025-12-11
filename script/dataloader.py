import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    
    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228
    elif dataset_name == 'pems03':
        n_vertex = 358
    elif dataset_name == 'pems04':
        n_vertex = 307
    elif dataset_name == 'pems07':
        n_vertex = 883
    elif dataset_name == 'pems08':
        n_vertex = 170

    return adj, n_vertex



def load_data(dataset_name, len_train, len_val,len_test):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:len_train + len_val + len_test]
    # print(len(train))
    return train, val, test


def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


if __name__ == '__main__':
    load_adj('pems04')[0]
    load_adj('pems03')[0]
    data = np.array(load_data('pems04',288,288,288)[0])
    data_transform(data, 12, 12, 'cuda')

    load_adj('pems04')[0]
    data1 = np.array(load_data('metr-la',288,288,288)[0])
    data = np.array(load_data('pems04',288,288,288)[0])
    data_transform(data, 12, 12, 'cuda')
    from sklearn import preprocessing
    zscore = preprocessing.StandardScaler() 
    train = zscore.fit_transform(data)
    train
