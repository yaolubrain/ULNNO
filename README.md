#Unsupervised Learning on Neural Network Outputs
This repo contains the experiment code in paper 

[Unsupervised Learning on Neural Network Outputs](http://arxiv.org/abs/1506.00990)

The paper presents a new zero-shot learning method, which achieves the state-of-the-art results in ImageNet fall2011.

The CNN model is GoogeLeNet with Caffe implementation. The image2hdf5 is from [Toronto Deep Learning](https://github.com/TorontoDeepLearning/convnet).

## Instructions
### download the following files from http://image-net.org/
- ILSVRC2012_img_train.tar (138G)
- ILSVRC2012_img_val.tar (6.3G)
- fall11_whole.tar (1.2T)

### prepare the images intro HDF5 files, use
- uncompress.sh
- correct_format.sh
- image2hdf5.sh

### compute the CNN outputs of GoogLeNet of the images, use
- caffe_outputs.py

### compute PCA and ICA on the CNN outputs, use
- cov.py
- whitening.py
- ica.py

### compute the MDS features of WordNet graph, use
- similarity_mat.py
- mds_distance_mat.m

### run zero-shot learning experiments, use
- imagenet_1k_21k_idx.py
- imagenet_zero_shot_unseen_wnids.py 
- make_zero_shot_mat.m
- zero_shot_random.py
- zero_shot_pca.py
- zero_shot_ica.py

## Questions
If you have any question regarding the code and the experiments, please contact me (yaolubrain@gmail.com). I would like to hear from you!
