function [model,opts]=LSSH_learn(Xtraining,Ytraining,hash_bits,opts)

% ||X-BS||^2 + rho*sum{|s|} + lambda*||Y - RS||^2+ mu*||T-AY||^2
% s.t. ||B||,||R||,||A|| <=1 
%   opts.mu = mu;
%   opts.rho = rho;
%   opts.lambda = lambda;
%   opts.maxOutIter = 20;
%Input:
%图像训练集，文本训练集，哈希码长度，参数：mu,lambda,rho,maxOutIter
%Output:
%图像的完全基集合B，稀疏稀疏S；文本的潜在语义A；


% Reference:
% Jile Zhou, GG Ding, Yuchen Guo
% "Latent Semantic Sparse Hashing for Cross-modal Similarity Search"
% ACM SIGIR 2014
% (Manuscript)
%
% Version1.0 -- Nov/2013
% Written by Jile Zhou (zhoujile539@gmail.com), Yuchen Guo (yuchen.w.guo@gmail.com)
%

warning off all
addpath('./Method-LSSH/SLEP_package_4.1/SLEP');
addpath('./Method-LSSH/SLEP_package_4.1/SLEP/functions/L1/L1R');
randNum=1;
%设置样本点的个数
nSamples = 10000;
if nSamples == size(Xtraining,1)
    training_id = 1:nSamples;
elseif (nSamples < size(Xtraining,1))
    fprintf('Training model by sampling %d points randomly\n',nSamples);
    training_id = randperm(size(Xtraining,1));
    training_id = training_id(1:nSamples);
else
    fprintf('Training model by using the whole %d training data\n',size(Xtraining,1));
    training_id = 1:size(Xtraining,1);
    nSamples = size(Xtraining, 1);
end
%%construct data
low_dim = hash_bits;
X = Xtraining(training_id, :)';
T = Ytraining(training_id, :)';

% 中心化数据，The data matrix is of size m x n
X = bsxfun(@minus,X, mean( (Xtraining'),2));% X in the paper
T = bsxfun(@minus,T, mean( (Ytraining'),2));% Y in the paper
%%% PCA %%%%%
% [U,S,V] = svd(cov(X'));
[U, ~] =  eigs(cov(X'), low_dim);%pca is for row data
PX = U(:,1:low_dim);%为了对数据进行主成份分析
% PX = eye(size(X, 1));
X = PX'* X;

 %U =  pca(T');%pca is for row data
%[U, ~] =  eigs(cov(T'),low_dim);%pca is for row data
%PT = U(:,1:low_dim);
PT = eye(size(T, 1));
T =PT'*T;%%T没发生变化。？为什么要做这一步处理呢？ 

%P = eye(size(X,1));
%% preprocessingData 对数据进行归一化处理
norm_y = sum(X.^2,1).^-.5;
for i = 1 : size(X,1)
    X(i,:) = X(i,:).*norm_y;
end

norm_t = sum(T.^2,1).^-.5;
for i = 1 : size(T,1)
    T(i,:) = T(i,:).*norm_t;
end

%% 初始化各个变量
%dim0 : sourse data dimention
%dim1 : sparse coding dimention
[dim0 num] = size(X);
dim1 = 512;%定义B，R的列数
randn('state',(randNum-1)*3+1);
A=randn(size(T, 1),hash_bits);       % the data matrix，
B=randn(dim0,dim1);
R=randn(hash_bits,dim1);

%% DFF获取参数
rho = opts.rho;  % the regularization parameter
mu = opts.mu;% mu in the paper
lambda = opts.lambda;% gama in the paper 
%rho=0.6;            % the regularization parameter

% it is a ratio between (0,1), if .rFlag=1

%----------------------- Set optional items ------------------------


% Starting point
opts.init=2;        % starting from a zero point

% termination criterion
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=30;   % maximum number of iterations

% normalization
opts.nFlag=0;       % without normalization

% regularization
opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
%opts.rsL2=0.01;     % the squared two norm term
%opts.lambda = 0.1;
t = 0 ; % out-loop itration
maxIter = opts.maxOutIter; % out-loop itration
opts.mFlag=0;       % treating it as compositive function
opts.lFlag=0;       % Nemirovski's line search

%----------------------- Run the code LeastR -----------------------
S = zeros(size(B,2),size(X,2));
%fprintf('ScmHashing initing: dim = %d, bits = %d, bases = %d, lambda = %s, sparse = %s, maxIter = %d \n',size(X,1),hash_bits,size(B,2),num2str(opts.lambda),num2str(rho),maxIter);
% addAttachedFiles(pool, LeastR)
for i = 1 : size(X,2)
    if (mod(i,1000)==0)
        fprintf('.');
    end
    [S(:,i), ~, ~]= LeastR(B, X(:,i), rho, opts);
end
fprintf('\n');
while t < maxIter
    %tic;
    Y = (A'*A + lambda/mu *eye(hash_bits))\(lambda/mu*R*S + A'*T);% A in the paper
    %Y = R*S*(1/opts.lambda);   
    A = l2ls_learn_basis_dual(T, Y, 1);% U in the paper
    %A = X*Y'/(Y*Y' + mu*eye(size(Y,1)));
for i = 1 : size(X,2)
    [S(:,i), ~, ~]= LeastR([B;sqrt(lambda) * R], [X(:,i);sqrt(lambda) * Y(:,i)], rho, opts);
 end
    
    B = l2ls_learn_basis_dual(X, S, 1);%文中的B
    R = l2ls_learn_basis_dual(Y, S, 1);%R in the paper
    
    sparse_error = sum(sum((X-B*S).^2));%利用稀疏编码求图像的稀疏表示
    sparse_embdding_error = sum(sum((Y-R*S).^2));%表示两种模态之间的关联
    matrix_factrozation_error = sum(sum((T-A*Y).^2));%利用矩阵分解求文本的潜在语义
    obj = sparse_error + rho* sum(sum(abs(S))) + opts.lambda*sparse_embdding_error + matrix_factrozation_error;% + mu*(sum(sum(B.^2))+sum(sum(R.^2))+sum(sum(A.^2)));
    t = t + 1;
    fprintf('%d/%d avgSpErr: %s, avgSpEmErr: %s, avgMfErr: %s, obj = %s, sparse = %.4f\n',t,maxIter,...
        num2str(sparse_error/size(X,2)),num2str(sparse_embdding_error/size(X,2)),num2str(matrix_factrozation_error/size(X,2)),num2str(obj), length(find(S == 0)) / nSamples / dim1);
    %toc;
    model.B=B;
    model.PX=PX;
    model.PT=PT;
    model.R=R;
    model.A=A;
    model.S=S;
end
