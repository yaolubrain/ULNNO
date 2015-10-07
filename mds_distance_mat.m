D = h5read('similarity_mat.h5','/path');
D = 1 - D;

tic
W = cmdscale(D);
toc

h5create('mds_distance_mat.h5','/W23',size(W));
h5write('mds_distance_mat.h5','/W23',W);

