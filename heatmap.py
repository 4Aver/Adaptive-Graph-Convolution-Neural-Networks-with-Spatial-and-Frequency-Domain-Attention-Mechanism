# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:41:07 2024

@author: Xue
"""

import numpy as np
import seaborn as sns
from scipy.sparse import coo_matrix,csr_matrix,csc_matrix
import scipy as sp
import matplotlib.pyplot as plt
 
# 创建一个随机的10x10矩阵作为示例数据
# data = np.random.rand(10, 10)
data = sp.sparse.load_npz('./data/pemsd7-m/adj.npz')
data.toarray()
data.shape
# 标准化数据（可选）
# data_normalized = (data - np.mean(data)) / np.std(data)
 
# 设置热力图的绘制
plt.figure(figsize=(10, 8))  # 调整画布大小
sns.heatmap(data.toarray(), 
            annot=False,          # 显示每个格子的数值
            fmt=".2f",           # 数值格式化，保留两位小数
            cmap='coolwarm',     # 颜色映射
            linewidths=0.5,      # 设置格子之间的线宽
            linecolor='black',   # 设置格子线的颜色
            cbar_kws={'label': 'Normalized Value'},  # 颜色条标签
            square=True)         # 强制将格子显示为正方形
 
# 添加标题和轴标签
plt.title('Heatmap of PEMSD7-M', fontsize=16)
plt.xlabel('Sensor ID')
plt.ylabel('Sensor ID')
 
# 显示图像
plt.tight_layout()  # 自动调整子图参数以填充图像
plt.show()