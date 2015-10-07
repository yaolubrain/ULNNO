import numpy as np
import h5py, sys, os, time, glob, caffe


caffe_path = '../caffe'
model_path = caffe_path + '/models/bvlc_googlenet/' 
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(net_fn, param_fn, caffe.TEST)

MB_SIZE = 128
CLASS_NUM = 1000
IM_NUM = 50000
HIDDEN_SIZE_1 = net.blobs['pool5/7x7_s1'].data.shape[1]
HIDDEN_SIZE_2 = net.blobs['loss3/classifier'].data.shape[1]

im_mean = np.float32([104, 117, 123]).reshape(1, 3, 1, 1)

for f in glob.glob("*[0-9].h5"):
    if os.path.isfile(f[0:9]+'_outputs.h5'):
        continue

    images_file = h5py.File(f, 'r')
    images = images_file['data']
    IM_NUM = images.shape[0]
    print f, IM_NUM

    outputs_prob = np.zeros((IM_NUM, CLASS_NUM), dtype=np.float32)
    outputs_logits = np.zeros((IM_NUM, HIDDEN_SIZE_2), dtype=np.float32)
    outputs_pool5 = np.zeros((IM_NUM, HIDDEN_SIZE_1), dtype=np.float32)
    index = 0
    while index < IM_NUM:
        chunk_size = MB_SIZE
        if index + MB_SIZE < IM_NUM:
            index += MB_SIZE
        else:
            chunk_size = IM_NUM - index
            index = IM_NUM

        im = images[index-chunk_size:index,:].reshape(chunk_size, 3, 224, 224)
        im = im[:,(2,1,0),:,:]   # swap RGB to BGR channels
        im = im - im_mean        # subtract mean value of each channel

        net.blobs['data'].reshape(chunk_size, 3, 224, 224)
        net.blobs['data'].data[...] = im
        net.forward()
        out_prob = net.blobs['prob'].data[:]
        out_logits = net.blobs['loss3/classifier'].data[:]
        out_pool5 = net.blobs['pool5/7x7_s1'].data[:]

        outputs_prob[index-chunk_size:index,:] = out_prob
        outputs_pool5[index-chunk_size:index,:] = np.squeeze(out_pool5)
        outputs_logits[index-chunk_size:index,:] = np.squeeze(out_logits)

    images_file.close()

    output_file = h5py.File(f[0:9]+'_outputs.h5', 'w')
    output_file.create_dataset('prob', data=outputs_prob)
    output_file.create_dataset('loss3/classifier', data=outputs_logits)
    output_file.create_dataset('pool5/7x7_s1', data=outputs_pool5)
    output_file.close()
