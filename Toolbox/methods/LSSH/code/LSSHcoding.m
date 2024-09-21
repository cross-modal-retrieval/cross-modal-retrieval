function [train_hash,test_hash_X, test_hash_T] = LSSHcoding(A,B,PX, PT,R,Xtraining, Ttraining, Xtest, Ttest, opts, hash_bits)

%
% Reference:
% Jile Zhou, GG Ding, Yuchen Guo
% "Latent Semantic Sparse Hashing for Cross-modal Similarity Search"
% ACM SIGIR 2014
% (Manuscript)
%
% Version1.0 -- Nov/2013
% Written by Jile Zhou (zhoujile539@gmail.com), Yuchen Guo (yuchen.w.guo@gmail.com)
%

train_data_X = preprocessingData(Xtraining',PX,mean( (Xtraining'),2));%对数据进行归一化处理
train_data_T = preprocessingData(Ttraining',PT,mean( (Ttraining'),2));

test_data_X = preprocessingData(Xtest',PX,mean( (Xtraining'),2));
test_data_T = preprocessingData(Ttest',PT,mean( (Ttraining'),2));

test_code_X = zeros(size(B,2),size(test_data_X,2));
train_code_X = zeros(size(B,2),size(train_data_X,2));


rho = opts.rho;% lambda in  the paper
for i = 1 : size(train_data_X,2)
    train_code_X(:,i) =  LeastR(B, train_data_X(:,i), rho, opts);
end
for i = 1 : size(test_data_X,2)
    test_code_X(:,i) =  LeastR(B, test_data_X(:,i), rho, opts);
end
test_hash_X = sign(R*test_code_X);

test_hash_T = sign((A'*A + opts.lambda/opts.mu*eye(size(A,2)))\A'*test_data_T);

train_hash = sign((A'*A + opts.lambda/opts.mu *eye(hash_bits))\(opts.lambda/opts.mu*R*train_code_X + A'*train_data_T));