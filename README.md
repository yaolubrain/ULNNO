#Unsupervised Learning on Neural Network Outputs
This repo contains the experiment code in paper 

[Unsupervised Learning on Neural Network Outputs](http://arxiv.org/abs/1506.00990)

The paper presents a new zero-shot learning method, which achieves the state-of-the-art results in ImageNet fall2011.

In order to reproduce the experiments, first you need to download the following files from http://image-net.org/

- ILSVRC2012_img_train.tar (138G)
- ILSVRC2012_img_val.tar (6.3G)
- fall11_whole.tar (1.2T)

Second, you need to prepare the images intro HDF5 files, using 
- uncompress.sh
- correct_format.sh
- image2hdf5.sh

Third, to compute the CNN outputs of GoogLeNet of the images, using
- caffe_outputs.py


