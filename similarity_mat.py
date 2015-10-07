import numpy as np
import h5py
from nltk.corpus import wordnet as wn

f_name = 'imagenet.synset.obtain_synset_list'
nodes_imagenet = open(f_name, 'r').read().splitlines()

def id2ss(ID):
    """Given a Synset ID (e.g. 01234567-n) return a synset"""
    return wn._synset_from_pos_and_offset(str(ID[-1:]), int(ID[:8]))

node_list = []
node_list.append('n04399382')  # this node 'teddy, teddy bear' is not in imagenet 21K 2011 fall
for node in nodes_imagenet:
    node_list.append(node)

D_path = np.zeros((len(node_list), len(node_list)))
D_lch = np.zeros((len(node_list), len(node_list)))
D_wup = np.zeros((len(node_list), len(node_list)))

for i in xrange(len(node_list)):
    id1 = node_list[i]
    n1 = id2ss(id1[1:]+'-n')
    print i, n1
    for j in xrange(len(node_list)):
        id2 = node_list[j]
        n2 = id2ss(id2[1:]+'-n')

        D_path[i,j] = wn.path_similarity(n1, n2)

f = h5py.File('similarity_mat.h5', 'w')
f.create_dataset('path', data=D_path)

