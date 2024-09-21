function [] = demo_SMFH(bits, dataname)
    bits = str2num(bits);

    data_dir = '../../Data';

    if ~exist(data_dir, 'dir')
        error('No such dir(%s)', fullfile(pwd, data_dir))
    end

    if ~exist('../result', 'dir')
        mkdir('../result')
    end

    addpath(data_dir);

    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    %% parameter settings
    alpha = 0.5;
    beta = 100;
    gamma = 1;
    lambda = 0.01;

    k_nn = 5; %5

    %% centralization
    %fprintf('centralizing data...\n');

    I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
    I_db = bsxfun(@minus, I_db, mean(I_tr, 1));
    I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));

    T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
    T_db = bsxfun(@minus, T_db, mean(T_tr, 1));
    T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

    %% mixed graph regularization term
    W_img = adjacency(I_tr, 'nn', k_nn); % model the intra-modal similarity in image modality
    W_txt = adjacency(T_tr, 'nn', k_nn); % model the intra-modal similarity in text modality
    W_inter = L_tr * L_tr'; % model the label cosistency between the image and text modality
    clear L_tr;

    [row, col] = size(I_tr);
    D_img = zeros(row, row);
    D_txt = zeros(row, row);
    D_inter = zeros(row, row);

    W_img(W_img ~= 0) = 1;
    W_txt(W_txt ~= 0) = 1;

    for i = 1:row
        D_img(i, i) = sum(W_img(i, :));
        D_txt(i, i) = sum(W_txt(i, :));
        D_inter(i, i) = sum(W_inter(i, :));
    end

    L_i = D_img - W_img;
    clear D_img W_img;

    L_t = D_txt - W_txt;
    clear D_txt W_txt;

    L_inter = D_inter - W_inter;
    clear D_inter W_inter;

    L = L_i + L_t + L_inter;
    clear L_i L_t L_inter;

    I_temp = I_tr';
    T_temp = T_tr';

    I_temp = bsxfun(@minus, I_temp, mean(I_temp, 2));
    T_temp = bsxfun(@minus, T_temp, mean(T_temp, 2));

    Im_te = (bsxfun(@minus, I_te', mean(I_tr', 2)))';
    Te_te = (bsxfun(@minus, T_te', mean(T_tr', 2)))';

    Im_db = (bsxfun(@minus, I_db', mean(I_tr', 2)))';
    Te_db = (bsxfun(@minus, T_db', mean(T_tr', 2)))';

    clear I_tr T_tr I_te T_te I_db T_db;

    %% solve the objective function
    [U1, U2, P1, P2, S] = solveSMFH(I_temp, T_temp, L, alpha, beta, gamma, lambda, bits);

    %% calculate hash codes
    Yi_te = sign((bsxfun(@minus, P1 * Im_te', mean(S, 2)))');
    Yi_db = sign((bsxfun(@minus, P1 * Im_db', mean(S, 2)))');

    Yt_te = sign((bsxfun(@minus, P2 * Te_te', mean(S, 2)))');
    Yt_db = sign((bsxfun(@minus, P2 * Te_db', mean(S, 2)))');

    %add by zhangzhen
    Yi_te = Yi_te > 0;
    Yi_db = Yi_db > 0;
    Yt_te = Yt_te > 0;
    Yt_db = Yt_db > 0;

    %% evaluate
    %fprintf('start evaluating...\n');
    cbTest_vis = compactbit(Yi_te(:, 1:bits));
    cbDb_text = compactbit(Yt_db(:, 1:bits));

    cbTest_text = compactbit(Yt_te(:, 1:bits));
    cbDb_vis = compactbit(Yi_db(:, 1:bits));

    hamm_T2I = hammingDist(cbTest_text, cbDb_vis)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    hamm_I2T = hammingDist(cbTest_vis, cbDb_text)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
