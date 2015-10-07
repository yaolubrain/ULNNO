imagenet_21k = open('imagenet.synset.obtain_synset_list', 'r').read().splitlines()
imagenet_21k.insert(0, 'n04399382') #this node 'teddy, teddy bear' is not in the imagenet 21K 2011 fall

imagenet_1k = open('synsets.txt', 'r').read().splitlines()
idx_list = open('imagenet_1k_21k_idx.txt', 'w')

for line in imagenet_1k:
    idx = imagenet_21k.index(line) 
    idx_list.write(str(idx) + '\n')
