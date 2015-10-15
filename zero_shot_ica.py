from __future__ import division
import numpy as np
import h5py
import sys
import time
import glob
from scipy.spatial.distance import cdist

DIM = 500

with h5py.File('zero_shot_mat_ica.h5', 'r') as f:
    sm_mean = f['sm_mean'][:]
    g_mean = f['g_mean'][:]
    W1 = f['W1'][:]
    P1 = f['P1'][:].T
    PW3 = f['PW3'][:].T
    PW23 = f['PW23'][:].T

print g_mean.shape, P1.shape, PW3.shape, PW23.shape

wnid_list_unseen = open('imagenet_zero_shot_unseen_wnids.txt', 'r').read().splitlines()
wnid_list_mixed = open('imagenet.synset.obtain_synset_list', 'r').read().splitlines()
wnid2idx_unseen = {}
wnid2idx_mixed = {}
for i in xrange(len(wnid_list_unseen)):
    wnid2idx_unseen[wnid_list_unseen[i]] = i
for i in xrange(len(wnid_list_mixed)):
    wnid2idx_mixed[wnid_list_mixed[i]] = i

correct_total_unseen = 0
correct_total_mixed = 0
im_num_total_unseen = 0
im_num_total_mixed = 0
for f in glob.glob("*.h5"):
    output_file = h5py.File(f, 'r')
    output = output_file['prob'][:]
    output_file.close()

    output = output - sm_mean

    g = np.dot(output, W1)
    g = g / np.sum(np.abs(g), 1)[:,np.newaxis]
    P1g = np.dot(g - g_mean, P1)

    im_num = output.shape[0]
    wnid = f[0:9]

    # unseen 
    if wnid2idx_unseen.has_key(wnid):
        class_idx = wnid2idx_unseen[wnid]
        D = cdist(P1g, PW3, 'cosine')
        pred_20 = np.argsort(D, axis=1)[:,:20]
        pred_10 = pred_20[:,:10]
        pred_5 = pred_20[:,:5]
        pred_2 = pred_20[:,:2]
        pred_1 = pred_20[:,:1]
        correct_20 = np.sum(pred_20 == class_idx)
        correct_10 = np.sum(pred_10 == class_idx)
        correct_5 = np.sum(pred_5 == class_idx)
        correct_2 = np.sum(pred_2 == class_idx)
        correct_1 = np.sum(pred_1 == class_idx)

        pred_file = open('pred_unseen_ica.txt', 'a')
        pred_file.write("%s %d %d %d %d %d %d\n" % (wnid, im_num, correct_1, correct_2, correct_5, correct_10, correct_20))
        pred_file.close()

        correct_total_unseen += correct_20
        im_num_total_unseen += im_num
        print correct_20, im_num,
    else: 
        print 0, 0,


    # mixed
    class_idx = wnid2idx_mixed[wnid]
    D = cdist(P1g, PW23, 'cosine')
    pred_20 = np.argsort(D, axis=1)[:,:20]
    pred_10 = pred_20[:,:10]
    pred_5 = pred_20[:,:5]
    pred_2 = pred_20[:,:2]
    pred_1 = pred_20[:,:1]
    correct_20 = np.sum(pred_20 == class_idx)
    correct_10 = np.sum(pred_10 == class_idx)
    correct_5 = np.sum(pred_5 == class_idx)
    correct_2 = np.sum(pred_2 == class_idx)
    correct_1 = np.sum(pred_1 == class_idx)

    pred_file = open('pred_mixed_ica.txt', 'a')
    pred_file.write("%s %d %d %d %d %d %d\n" % (wnid, im_num, correct_1, correct_2, correct_5, correct_10, correct_20))
    pred_file.close()

    correct_total_mixed += correct_20
    im_num_total_mixed += im_num

    print correct_20, im_num,      

    if im_num_total_unseen != 0:
        print correct_total_unseen / im_num_total_unseen,
    else:
        print 0.0,

    if im_num_total_mixed != 0:
        print correct_total_mixed / im_num_total_mixed

    


