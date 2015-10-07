clear all
setenv('LC_ALL','C')

n = 1000;

wnid2idx = containers.Map;
wnids_file = fopen('distance_mat_wordnet_nodes_1K.txt','r');
wnids = textscan(wnids_file,'%s\n');
fclose(wnids_file);
for i = 1:1000
    wnid2idx(wnids{1}{i}) = i;
end

gt = importdata('../caffe/data/ilsvrc12/val.txt');
gt = gt.data + 1;

idx_seen = importdata('imagenet_1k_21k_idx.txt');
idx_seen = idx_seen + 1;
idx_unseen = setdiff([1:21842]',idx_seen);



% Random 
%W1 = randn(size(W1));
%W1 = randn(1000,1000);
%[U,D,V] = svd(W1);
%W1 = U*V';
%W1 = W1(1:500,:);

% PCA
%C = h5read('cov.h5','/cov');
%C = double(C);
%[E, D] = eig(C);
%[~,order] = sort(diag(-D));
%E = E(:,order);
% W1 = E(:,1:500)';

% ICA
U = h5read('ica.h5','/U')';
V = h5read('ica.h5','/V')';
U = double(U);
V = double(V);
W1 = V*U;
W1 = bsxfun(@rdivide, W1, sqrt(sum(W1.^2,2)));

sm_mean = 0.001*ones(1000,1);

M = eye(n);

G = g(W1*bsxfun(@minus,M,sm_mean))';

W23 = h5read('mds_distance_mat.h5','/W23');
W2 = W23(idx_seen,:);
W3 = W23(idx_unseen,:);

[P1,P2,r,PW1,PW2] = canoncorr(G,W2);

PW3 = (bsxfun(@minus,W3,mean(W2,1))*P2);
PW23 = (bsxfun(@minus,W23(2:end,:),mean(W2,1))*P2);

f_name = 'zero_shot_mat_ica_500.h5';
h5create(f_name,'/g_mean',size(mean(G,1)'))
h5create(f_name,'/sm_mean',size(sm_mean))
h5create(f_name,'/W1',size(W1))
h5create(f_name,'/P1',size(P1))
h5create(f_name,'/PW2',size(PW2))
h5create(f_name,'/PW3',size(PW3))
h5create(f_name,'/PW23',size(PW23))
h5write(f_name,'/g_mean',mean(G,1)')
h5write(f_name,'/sm_mean',sm_mean)
h5write(f_name,'/W1',W1)
h5write(f_name,'/P1',P1)
h5write(f_name,'/PW2',PW2)
h5write(f_name,'/PW3',PW3)
h5write(f_name,'/PW23',PW23)
    
