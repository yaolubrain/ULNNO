import numpy as np
import h5py
import sys
import time
from numpy import linalg as la

IM_NUM = 1281167
MB_SIZE = 100
CLASS_NUM = 1000

output= h5py.File('imagenet_1k_outputs_googlenet.h5', 'r')

mean = np.zeros((1, CLASS_NUM))
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

    mean += 1.0/IM_NUM * np.sum(X, 0)
    print np.sum(X)


cov = np.zeros((CLASS_NUM, CLASS_NUM))
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

    X -= mean

    cov += 1.0/IM_NUM * np.dot(X.T, X)

output.close()

mean_file = h5py.File('mean.h5','w')
mean_file.create_dataset('mean', data=mean)
mean_file.close()

cov_file = h5py.File('cov.h5','w')
cov_file.create_dataset('cov', data=cov)
cov_file.close()





