import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import cvxpy as cvx
import torch
import scipy.optimize as spopt
import scipy.fftpack as spfft
from sklearn.linear_model import Lasso
import matplotlib
import random
matplotlib.use('TkAgg') # for plot not responding

# load train and test data to tensor
# train_data = 60000 , test_data = 10000
y = torch.load('svdb/test_t02mc6.pt')
datas = y[0]
label = y[1]

n = 1280
m = 128*8

# Phi = np.random.binomial(size=m * n, n=1, p=0.5)  # 베르누이 랜덤 행렬
Phi = np.random.normal(loc=0.0, scale=1.0, size=m*n) # 가우시안 랜덤 행렬
Phi = Phi.reshape(-1, n)
A = spfft.idct(np.identity(n), norm='ortho', axis=0)
Theta = np.matmul(Phi,A)

s_train = torch.zeros([10000,1280,1])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


for i in range(4040):
    data = datas[i,:]
    data = data.numpy()

    y2 = np.matmul(Phi,data)
    # do L1 optimization
    result = Lasso(alpha=0.001)
    result.fit(Theta, y2)
    s = np.array(result.coef_)
    s = np.expand_dims(s,-1)

    s = abs(s)
    s = scaler.fit_transform(s)
    s *= 128
    s = torch.from_numpy(s)

    s_train[i] = s

data_save = (s_train,label)
torch.save(data_save, 'svdb/test_10_128.pt')
