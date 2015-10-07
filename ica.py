import numpy as np
import h5py
import sys
import time
import re
import numpy.linalg as la
import theano
import theano.tensor as T

IM_NUM = 1281167
CLASS_NUM = 1000
MB_SIZE = 500
DIM = 900
ITER = 1000

images = h5py.File('whitening.h5', 'r')['feature']
U = h5py.File('whitening.h5', 'r')['U'][:]
ica_file = h5py.File('ica.h5', 'w')
ica_file.create_dataset('V', (DIM, DIM))
ica_file.create_dataset('U', data=U[:DIM,:])
ica_file.close()

v = np.random.randn(DIM, DIM)
v1, _, v2 = la.svd(v)
v = np.dot(v1, v2.T)
V = theano.shared(np.asarray(v, dtype=np.float32), borrow=True)
Z = T.matrix()
I = T.eye(DIM)
LR = T.scalar(dtype='float32')
MB = T.scalar(dtype='float32')
gY = 1.0 - 2.0/(T.exp(2*(T.dot(V,Z))) + 1.0)
#gY = T.tanh(T.dot(V,Z))
update = {V: V - LR/MB*T.dot(gY,Z.T) + 0.5*T.dot(I - T.dot(V,V.T),V)}
ICA = theano.function([Z, LR, MB],updates=update)

lr = 0.005

for i in xrange(ITER):
    perm = np.random.permutation(IM_NUM)
    v_prev = v
    index = 0
    while index < IM_NUM:
        chunk_size = MB_SIZE
        if index + MB_SIZE < IM_NUM:
            index += MB_SIZE
        else:
            chunk_size = IM_NUM - index
            index = IM_NUM

        data_idx = np.sort(perm[index-chunk_size:index])        
        z = images[data_idx,:DIM].T

        ICA(z, lr, chunk_size)

    if i % 10 == 0:
        lr = lr / 2

    v = V.get_value()
    print np.sum(np.abs(v - v_prev)), np.sum(np.square(v))

    ica_file = h5py.File('ica.h5', 'r+')
    ica_file['V'][:] = v
    ica_file.close()

