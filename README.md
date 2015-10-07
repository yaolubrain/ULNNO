#Unsupervised Learning on Neural Network Outputs
This repo contains the experiment code in paper 

[Unsupervised Learning on Neural Network Outputs](http://arxiv.org/abs/1506.00990)

The paper presents a new zero-shot learning method, which achieves the state-of-the-art results in ImageNet fall2011.

## Instructions
### Download the following files from http://image-net.org/
- ILSVRC2012_img_train.tar (138G)
- ILSVRC2012_img_val.tar (6.3G)
- fall11_whole.tar (1.2T)

### prepare the images intro HDF5 files, use
- uncompress.sh
- correct_format.sh
- image2hdf5.sh

3. compute the CNN outputs of GoogLeNet of the images, use
- caffe_outputs.py

4. compute PCA and ICA on the CNN outputs, use
- cov.py
- whitening.py
- ica.py

5.

