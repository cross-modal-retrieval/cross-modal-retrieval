
% Reference:
% Jile Zhou, GG Ding, Yuchen Guo
% "Latent Semantic Sparse Hashing for Cross-modal Similarity Search"
% ACM SIGIR 2014
% (Manuscript)
%
% Version1.0 -- Nov/2013
% Written by Yuchen Guo (yuchen.w.guo@gmail.com)
%

clear;
clc;
load labelme;
if matlabpool('size') <=0 
    matlabpool;
end
run = 1;
map = zeros(run,2);
nbits = [8, 16, 24, 32, 48, 64, 96, 128];
bits = 64;
mu = 0.5;
rho = 0.2;
lambda = 1;

I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

for b = 2 : 2 : 8
    tic;
    bits = nbits(1, b);
    fprintf('bits = %d, rho = %.4f, lambda = %.4f, mu = %.4f\r\n', bits, rho, lambda, mu);

    for i = 1 : run

    % construct training set

        I_temp = I_tr';
        T_temp = T_tr';
        [row, col]= size(I_temp);
        [rowt, colt] = size(T_temp);


        I_temp = bsxfun(@minus,I_temp , mean(I_temp,2));
        T_temp = bsxfun(@minus,T_temp, mean(T_temp,2));
        Im_te = (bsxfun(@minus, I_te', mean(I_tr', 2)))';
        Te_te = (bsxfun(@minus, T_te', mean(T_tr', 2)))';

        opts = [];
        opts.mu = mu;
        opts.rho = rho;
        opts.lambda = lambda;
        opts.maxOutIter = 20;
        [B,PX,PT,R,A,S,opts]= solveLSSH(I_temp',T_temp',bits,opts);
        [train_hash,test_hash_I, test_hash_T] = LSSHcoding(A, B,PX, PT,R,I_temp',T_temp',Im_te,Te_te,opts, bits);
       
        sim = train_hash' * test_hash_I;
        MAP = mAP(sim,L_tr,L_te, 0);
        fprintf('image to text: %.4f\n', MAP);

        
        sim = train_hash' * test_hash_T;
        MAP = mAP(sim,L_tr,L_te, 0);
        fprintf('text to image: %.4f\r\n', MAP);
    end
    toc;
end
if matlabpool('size') > 0 
    matlabpool close;
end

% heterogeneous transfer hashing 
% 
% clear;
% clc;
% load labelme;
% if matlabpool('size') <=0 
%     matlabpool;
% end
% run = 1;
% map = zeros(run,2);
% nbits = [8, 16, 24, 32, 48, 64, 96, 128];
% bits = 64;
% mu = 0.5;
% rho = 0.2;
% lambda = 1;
% nmu = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100];
% nlambda = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100];
% I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
% I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
% T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
% T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));
% fid = fopen('result/LM_lambda.txt', 'w');
% 
% for l = 1 : 1 : 13
%     tic;
%     bits = nbits(1, b);
%     mu = nmu(m);
%     lambda = nlambda(l);
%     fprintf(fid, 'bits = %d, rho = %.4f, lambda = %.4f, mu = %.4f\r\n', bits, rho, lambda, mu);
%     fprintf('bits = %d, rho = %.4f, lambda = %.4f, mu = %.4f\r\n', bits, rho, lambda, mu);
% 
%     for i = 1 : run
% 
%     construct training set
% 
%         I_temp = I_tr';
%         T_temp = T_tr';
%         [row, col]= size(I_temp);
%         [rowt, colt] = size(T_temp);
% 
% 
%         I_temp = bsxfun(@minus,I_temp , mean(I_temp,2));
%         T_temp = bsxfun(@minus,T_temp, mean(T_temp,2));
%         Im_te = (bsxfun(@minus, I_te', mean(I_tr', 2)))';
%         Te_te = (bsxfun(@minus, T_te', mean(T_tr', 2)))';
% 
%         opts = [];
%         opts.mu = mu;
%         opts.rho = rho;
%         opts.lambda = lambda;
%         opts.maxOutIter = 20;
%         [B,PX,PT,R,A,S,opts]= sparse_matrix(I_temp',T_temp',bits,opts);
%         [train_hash,test_hash_I, test_hash_T] = sparse_matrix_coding(A, B,PX, PT,R,I_temp',T_temp',Im_te,Te_te,opts, bits);
%         [train_hash_T,test_hash_T] = sparse_matrix_coding(A, B,PX,PT,R,I_temp',T_temp',Im_te',Te_te',opts, 1);
% 
% 
%         % identical Y for image and text
%         [X1, X2, Wi, Wt, Y] = solveHTH(I_temp, T_temp, lambda, mu, gamma, bits);
%         Y_total = (X1' * X1 + lambda * X2' * X2 + 2 * mu * eye(bits) + gamma * eye(bits)) \ (X1' * I_tr' + lambda * X2' * T_tr' + mu * (Wi * I_tr' + Wt * T_tr'));
%         Y = rand(bits, col);
%         Yi_tr = sign((bsxfun(@minus, Y_total , mean(Y,2)))');
%         Yi_te = sign((bsxfun(@minus,Wi * I_te' , mean(Y,2)))');
%         Yt_tr = sign((bsxfun(@minus, Y_total , mean(Y,2)))');
%         Yt_te = sign((bsxfun(@minus,Wt * T_te' , mean(Y,2)))');
% 
%         different Y for image and text
%         [X1, X2, Yi] = solveSMH(I_temp, T_temp, 1, 0.2, 0.01, bits);
%         Wi = Yi * I_temp' / (I_temp * I_temp' + eye(row));
%         [X1, X2, Yt] = solveSMH(T_temp, I_temp, 1, 0.2, 0.01, bits);
%         Wt = Yt * T_temp' / (T_temp * T_temp' + eye(rowt));
%         Yi_tr = sign((bsxfun(@minus, Yi , mean(Yi,2)))');
%         Yi_te = sign((bsxfun(@minus,Wi * I_te' , mean(Yi,2)))');
%         Yt_tr = sign((bsxfun(@minus, Yt , mean(Yt,2)))');
%         Yt_te = sign((bsxfun(@minus,Wt * T_te' , mean(Yt,2)))');
%         sim = train_hash' * test_hash_I;
%         map(i, 1) = mAP(sim,L_tr,L_te, 0);
%         MAP = map(i, 1);
%         fprintf(fid, 'image to text: %.4f\r\n', MAP);
%         fprintf('image to text: %.4f\n', MAP);
%         precision = topNprecision(sim, L_tr, L_te, 2000);
%         [rprecision, rrecall] = precision_recall_radius(sim, L_tr, L_te, bits);
%         [sprecision, srecall] = precision_recall_standard(sim, L_tr, L_te, bits);
%         str = strcat('result/', 'SCMH_i2t_', num2str(bits), '_result');
%         save (str, 'MAP', 'precision', 'rprecision', 'rrecall', 'sprecision', 'srecall');
%         sim = train_hash' * test_hash_T;
%         map(i, 2) = mAP(sim,L_tr,L_te, 0);
%         MAP = map(i, 2);
%         fprintf(fid, 'text to image: %.4f\r\n', MAP);
%         fprintf('text to image: %.4f\n', MAP);
%         precision = topNprecision(sim, L_tr, L_te, 2000);
%         [rprecision, rrecall] = precision_recall_radius(sim, L_tr, L_te, bits);
%         [sprecision, srecall] = precision_recall_standard(sim, L_tr, L_te, bits);
%         str = strcat('result/', 'SCMH_t2i_', num2str(bits), '_result');
%         save (str, 'MAP', 'precision', 'rprecision', 'rrecall', 'sprecision', 'srecall');
% 
% 
% 
%         simii = train_hash_I' * test_hash_I;
%         simti = train_hash_I' * test_hash_T;
%         simit = train_hash_I' * test_hash_I;
%         simtt = train_hash_I' * test_hash_T;
%         map(i, 1) = mAP(simii,L_tr,L_te);
%         map(i, 2) = mAP(simti,L_tr,L_te);
%         map(i, 3) = mAP(simit,L_tr,L_te);
%         map(i, 4) = mAP(simtt,L_tr,L_te);
%         fprintf('run %d, map = %.4f, %.4f, %.4f, %.4f\n', i, map(i, 1), map(i, 2), map(i, 3), map(i, 4));
%     end
%     mean(map);
%     fprintf('\nlambda = %.4f, bits = %d, mu = %d, gamma = %.4f\n', lambda, bits, mu, gamma);
%     fprintf('average map over %d runs for ImageQueryOnImageDB: %.4f\n', run, mean(map( : , 1)));
%     fprintf('average map over %d runs for TextQueryOnImageDB: %.4f\n', run, mean(map( : , 2)));
%     fprintf('average map over %d runs for ImageQueryOnTextDB: %.4f\n', run, mean(map( : , 1)));
%     fprintf('average map over %d runs for TextQueryOnTextDB: %.4f\n', run, mean(map( : , 4)));
%     toc;
% end
% if matlabpool('size') > 0 
%     matlabpool close;
% end
% fclose(fid);