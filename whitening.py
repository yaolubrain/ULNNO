import numpy as np
import h5py
import sys
import time
from numpy import linalg as la

IM_NUM = 1281167
MB_SIZE = 128
DIM = 999
CLASS_NUM = 1000

output= h5py.File('imagenet_1k_outputs_googlenet.h5', 'r')
mean = h5py.File('mean.h5','r')['mean'][:]
cov = h5py.File('cov.h5','r')['cov'][:]

D, E = la.eig(cov)
idx = D.argsort()[::-1]
d = D[idx]**(-0.5)
for i in xrange(len(d)):
    if np.isnan(d[i]):
        d[i] = 0
D = np.diag(d)
E = E[:,idx]
U = np.dot(D, E.T)
feat_file = h5py.File('whitening.h5', 'w')
feat_file.create_dataset('feature', (IM_NUM, DIM))
feat_file.create_dataset('U', data=U[:DIM,:])
feat_file.close()

index = 0 
while index < IM_NUM:
    chunk_size = MB_SIZE
    if index + MB_SIZE < IM_NUM:
        index += MB_SIZE
    else:
        chunk_size = IM_NUM - index
        index = IM_NUM

    print index, IM_NUM, chunk_size

    X = output['loss3/classifier'][index-chunk_size:index,:]
    X = X - np.min(X, 1)[:, np.newaxis]
    X = X / np.sum(X, 1)[:, np.newaxis]

    X = X - mean

    feat = np.dot(X, U[:DIM,:].T)
    feat_file = h5py.File('whitening.h5', 'r+')['feature']
    feat_file[index-chunk_size:index,:] = feat 
