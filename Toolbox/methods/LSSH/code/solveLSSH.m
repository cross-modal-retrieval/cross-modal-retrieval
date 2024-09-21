function [B, PX, PT, R, A, S, opts] = solveLSSH(Xtraining, Ytraining, hash_bits, opts)
    % ||X-BS||^2 + rho*sum{|s|} + lambda*||Y - RS||^2+ mu*||T-AY||^2
    % s.t. ||B||,||R||,||A|| <=1
    %   opts.mu = mu;
    %   opts.rho = rho;
    %   opts.lambda = lambda;
    %   opts.maxOutIter = 20;
    %Input:
    %ͼ��ѵ�������ı�ѵ��������ϣ�볤�ȣ�������mu,lambda,rho,maxOutIter
    %Output:
    %ͼ�����ȫ������B��ϡ��ϡ��S���ı���Ǳ������A��

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
    addpath('./SLEP_package_4.1/SLEP/');
    addpath('./SLEP_package_4.1/SLEP/functions/L1/L1R');
    randNum = 1;
    %����������ĸ���
    nSamples = 10000;

    if nSamples == size(Xtraining, 1)
        training_id = 1:nSamples;
    elseif (nSamples < size(Xtraining, 1))
        fprintf('Training model by sampling %d points randomly\n', nSamples);
        training_id = randperm(size(Xtraining, 1));
        training_id = training_id(1:nSamples);
    else
        fprintf('Training model by using the whole %d training data\n', size(Xtraining, 1));
        training_id = 1:size(Xtraining, 1);
        nSamples = size(Xtraining, 1);
    end

    %%construct data
    low_dim = hash_bits;
    X = Xtraining(training_id, :)';
    T = Ytraining(training_id, :)';

    % ���Ļ����ݣ�The data matrix is of size m x n
    X = bsxfun(@minus, X, mean((Xtraining'), 2)); % X in the paper
    T = bsxfun(@minus, T, mean((Ytraining'), 2)); % Y in the paper
    %%% PCA %%%%%
    % [U,S,V] = svd(cov(X'));
    [U, ~] = eigs(cov(X'), low_dim); %pca is for row data
    PX = U(:, 1:low_dim); %Ϊ�˶����ݽ������ɷݷ���
    % PX = eye(size(X, 1));
    X = PX' * X;

    %U =  pca(T');%pca is for row data
    %[U, ~] =  eigs(cov(T'), low_dim);%pca is for row data
    %PT = U(:,1:low_dim);
    PT = eye(size(T, 1));

    T = PT' * T; % %Tû�����仯����ΪʲôҪ����һ�������أ�

    %P = eye(size(X,1));
    %% preprocessingData �����ݽ��й�һ������
    norm_y = sum(X.^2, 1).^ - .5;

    for i = 1:size(X, 1)
        X(i, :) = X(i, :) .* norm_y;
    end

    norm_t = sum(T.^2, 1).^ - .5;

    for i = 1:size(T, 1)
        T(i, :) = T(i, :) .* norm_t;
    end

    %% ��ʼ����������
    %dim0 : sourse data dimention
    %dim1 : sparse coding dimention
    [dim0 num] = size(X);
    dim1 = 512; %����B��R������
    randn('state', (randNum - 1) * 3 + 1);
    A = randn(size(T, 1), hash_bits); % the data matrix��
    B = randn(dim0, dim1);
    R = randn(hash_bits, dim1);

    %% DFF��ȡ����
    rho = opts.rho; % the regularization parameter
    mu = opts.mu; % mu in the paper
    lambda = opts.lambda; % gama in the paper
    %rho=0.6;            % the regularization parameter

    % it is a ratio between (0,1), if .rFlag=1

    %----------------------- Set optional items ------------------------

    % Starting point
    opts.init = 2; % starting from a zero point

    % termination criterion
    opts.tFlag = 5; % run .maxIter iterations
    opts.maxIter = 30; % maximum number of iterations

    % normalization
    opts.nFlag = 0; % without normalization

    % regularization
    opts.rFlag = 1; % the input parameter 'rho' is a ratio in (0, 1)
    %opts.rsL2=0.01;     % the squared two norm term
    %opts.lambda = 0.1;
    t = 0; % out-loop itration
    maxIter = opts.maxOutIter; % out-loop itration
    opts.mFlag = 0; % treating it as compositive function
    opts.lFlag = 0; % Nemirovski's line search

    %----------------------- Run the code LeastR -----------------------
    S = zeros(size(B, 2), size(X, 2));
    %fprintf('ScmHashing initing: dim = %d, bits = %d, bases = %d, lambda = %s, sparse = %s, maxIter = %d \n',size(X,1),hash_bits,size(B,2),num2str(opts.lambda),num2str(rho),maxIter);
    % addAttachedFiles(pool, LeastR)
    for i = 1:size(X, 2)
        %if (mod(i,1000)==0)
        %    fprintf('.');
        %end
        [S(:, i), ~, ~] = LeastR(B, X(:, i), rho, opts);
    end

    %fprintf('\n');
    while t < maxIter
        %tic;
        Y = (A' * A + lambda / mu * eye(hash_bits)) \ (lambda / mu * R * S + A' * T); % A in the paper
        %Y = R*S*(1/opts.lambda);

        A = l2ls_learn_basis_dual(T, Y, 1); % U in the paper
        %A = X*Y'/(Y*Y' + mu*eye(size(Y,1)));
        for i = 1:size(X, 2)
            [S(:, i), ~, ~] = LeastR([B; sqrt(lambda) * R], [X(:, i); sqrt(lambda) * Y(:, i)], rho, opts);
        end

        B = l2ls_learn_basis_dual(X, S, 1); %���е�B
        R = l2ls_learn_basis_dual(Y, S, 1); %R in the paper

        sparse_error = sum(sum((X - B * S).^2)); %����ϡ�������ͼ���ϡ���ʾ
        sparse_embdding_error = sum(sum((Y - R * S).^2)); %��ʾ����ģ̬֮��Ĺ���
        matrix_factrozation_error = sum(sum((T - A * Y).^2)); %���þ���ֽ����ı���Ǳ������
        obj = sparse_error + rho * sum(sum(abs(S))) + opts.lambda * sparse_embdding_error + matrix_factrozation_error; % + mu*(sum(sum(B.^2))+sum(sum(R.^2))+sum(sum(A.^2)));
        t = t + 1;
        %fprintf('%d/%d avgSpErr: %s, avgSpEmErr: %s, avgMfErr: %s, obj = %s, sparse = %.4f\n',t,maxIter,...
        %    num2str(sparse_error/size(X,2)),num2str(sparse_embdding_error/size(X,2)),num2str(matrix_factrozation_error/size(X,2)),num2str(obj), length(find(S == 0)) / nSamples / dim1);
        %toc;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% coding

    %save('gist_SpH_matrix_1024_itr25_64_lambda_01','train_hash','test_hash');
