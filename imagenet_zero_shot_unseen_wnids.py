imagenet_21k = open('imagenet.synset.obtain_synset_list', 'r').read().splitlines()
imagenet_1k = open('distance_mat_wordnet_nodes_1K.txt', 'r').read().splitlines()
idx_list = open('imagenet_zero_shot_unseen_wnids.txt', 'w')

for line in imagenet_21k:
    if not line in imagenet_1k:
        idx_list.write(line+'\n')

