function [] = demo_SePH(bits, dataname)
    addpath(genpath('markSchmidt/')); % Refer to: http://www.cs.ubc.ca/~schmidtm/Software/code.html
    addpath('./utils')

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

    % Model parameters
    Model.alpha = 1e-2;

    v = 2;
    viewsName = {'Image', 'Text'};
    % retrieval set
    dbXs = cell(1, v);
    dbXs{1} = I_db;
    dbXs{2} = T_db;
    % train set
    trXs = cell(1, v);
    trXs{1} = I_tr;
    trXs{2} = T_tr;
    % query set
    tsXs = cell(1, v);
    tsXs{1} = I_te;
    tsXs{2} = T_te;

    tr_num = size(I_tr, 1);
    clear I_tr T_tr I_te T_te I_db T_db;
    meanV = cell(1, 2);
    %matlabpool 5;
    for i = 1:v
        meanV{i} = mean(trXs{i}, 1);
        trXs{i} = bsxfun(@minus, trXs{i}, meanV{i});
    end

    % Calculation of P for supervised learning (normalized cosine similarity)
    P = zeros(tr_num, tr_num);
    L_sample = L_tr; %(sampleInds, :);

    if size(L_sample, 2) > 1
        num1 = 1 ./ sqrt(sum(L_sample, 2)); % L_sample should be in {0, 1}
        num1(isinf(num1) | isnan(num1)) = 1;
        L_sample = diag(num1) * L_sample;

        P = L_sample * L_sample';
    else

        for ti = 1:tr_num
            P(ti, :) = double(L_sample == L_sample(ti))';
        end

    end

    L_sample = L_tr; %(sampleInds, :); % recover

    % Training & Testing
    teN = size(L_te, 1);
    dbN = size(L_db, 1);

    %trainMAPs = zeros(bitN, runtimes);
    ydata = minKLD(Model.alpha / bits / tr_num, P, bits);
    trainH = sign(ydata);

    % RBF kernel
    z = trXs{1} * trXs{1}';
    z = repmat(diag(z), 1, tr_num) + repmat(diag(z)', tr_num, 1) - 2 * z;
    k1 = {};
    k1.type = 0;
    k1.param = mean(z(:));

    z = trXs{2} * trXs{2}';
    z = repmat(diag(z), 1, tr_num) + repmat(diag(z)', tr_num, 1) - 2 * z;
    k2 = {};
    k2.type = 0;
    k2.param = mean(z(:));
    % p(c_k=1) and p(c_k=-1)
    learntP = [sum(trainH == 1, 1) / size(trainH, 1); sum(trainH == -1, 1) / size(trainH, 1); ];
    % Kernel logistic regression��developed by mark schimidt
    kernelSampleNum = 500;
    %kernelSampleNum = kernelSamps(di);
    % if kernelSampleNum > tr_num / 2
    %     continue;
    % end

    zXs = cell(1, v);
    % Kmeans sampling
    sampleType = 'km';
    opts = statset('Display', 'off', 'MaxIter', 100);

    [INX, K] = kmeans(trXs{1}, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
    zXs{1} = K;

    [INX, K] = kmeans(trXs{2}, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
    zXs{2} = K;

    % Kernel Matrices
    K01 = kernelMatrix(zXs{1}, zXs{1}, k1);
    K02 = kernelMatrix(zXs{2}, zXs{2}, k2);
    K1 = kernelMatrix(trXs{1}, zXs{1}, k1);
    K2 = kernelMatrix(trXs{2}, zXs{2}, k2);
    % fprintf('Train time [%s] bit [%d] [%.2f]\n', datasets{di}, bit, tr_time);
    % train end
    for i = 1:v
        tsXs{i} = bsxfun(@minus, tsXs{i}, meanV{i});
        dbXs{i} = bsxfun(@minus, dbXs{i}, meanV{i});
    end

    teK1 = kernelMatrix(tsXs{1}, zXs{1}, k1);
    teK2 = kernelMatrix(tsXs{2}, zXs{2}, k2);
    dbK1 = kernelMatrix(dbXs{1}, zXs{1}, k1);
    dbK2 = kernelMatrix(dbXs{2}, zXs{2}, k2);

    % Hash Codes for Retrieval Set and Query Set
    B1 = zeros(size(L_db, 1), bits); % Unique Hash Codes for Both Views of Retrieval Set
    B_te_txt = zeros(teN, bits); % Hash Codes for Image View of Query Set
    B_te_img = zeros(teN, bits); % Hash Codes for Text View of Query Set
    B_db_txt = zeros(dbN, bits); % Hash Codes for Image View of Query Set
    B_db_img = zeros(dbN, bits); % Hash Codes for Text View of Query Set

    options = {};
    options.Display = 'final';
    C = 0.01; % Weight for Regularization. We Found that 1e-2 is Good Enough.
    % Test time start
    % Kernel logistic regression��developed by mark schimidt
    for b = 1:bits
        tH = trainH(:, b);
        ppos = 1 / learntP(1, b); % 1/p(c_k=1)
        pneg = 1 / learntP(2, b); % 1/p(c_k=-1)
        ppos(isinf(ppos) | isnan(ppos)) = 1;
        ppos(isinf(pneg) | isnan(pneg)) = 1;

        % View 1 (Image View)
        funObj = @(u)LogisticLoss(u, K1, tH);
        w = minFunc(@penalizedKernelL2, zeros(size(K01, 1), 1), options, K01, funObj, C);
        %for test
        B_te_txt(:, b) = sign(teK1 * w);
        %for db
        B_db_txt(:, b) = sign(dbK1 * w);

        % View 2 (Text View)
        funObj = @(u)LogisticLoss(u, K2, tH);
        w = minFunc(@penalizedKernelL2, zeros(size(K02, 1), 1), options, K02, funObj, C);
        %for test
        B_te_img(:, b) = sign(teK2 * w);
        %for db
        B_db_img(:, b) = sign(dbK2 * w);
    end

    B_te_txt = bitCompact(sign(B_te_txt) >= 0);
    B_te_img = bitCompact(sign(B_te_img) >= 0);
    B_db_txt = bitCompact(sign(B_db_txt) >= 0);
    B_db_img = bitCompact(sign(B_db_img) >= 0);

    hamm_T2I = double(HammingDist(B_te_txt, B_db_img))';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    hamm_I2T = double(HammingDist(B_te_img, B_db_txt))';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
