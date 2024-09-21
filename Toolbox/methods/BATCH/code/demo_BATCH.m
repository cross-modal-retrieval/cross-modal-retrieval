function [] = demo_BATCH(bits, dataname)

    addpath('./codes/BATCH/');
    addpath('./utils');
    addpath('./utils/tool/');
    data_dir = '../../Data';

    if ~exist(data_dir, 'dir')
        error('No such dir(%s)', fullfile(pwd, data_dir))
    end

    if ~exist('../result', 'dir')
        mkdir('../result')
    end

    addpath(data_dir);
    bits = str2num(bits);

    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    train_dataX = I_tr;
    train_dataT = T_tr;
    query_dataX = I_te;
    query_dataT = T_te;
    retrieval_dataX = I_db;
    retrieval_dataT = T_db;

    L_train = L_tr;

    param.nbits = bits;

    %% Kernel representation
    param.nXanchors = 500; param.nYanchors = 1000;

    if 1
        anchor_idx = randsample(size(train_dataX, 1), param.nXanchors);
        XAnchors = train_dataX(anchor_idx, :);
        anchor_idx = randsample(size(train_dataT, 1), param.nYanchors);
        YAnchors = train_dataT(anchor_idx, :);
    else
        [~, XAnchors] = litekmeans(train_dataX, param.nXanchors, 'MaxIter', 30);
        [~, YAnchors] = litekmeans(train_dataT, param.nYanchors, 'MaxIter', 30);
    end

    [XKTrain] = Kernel_Feature_train(train_dataX, XAnchors);
    [YKTrain] = Kernel_Feature_train(train_dataT, YAnchors);

    BATCHparam = param;
    BATCHparam.eta1 = 0.05; BATCHparam.eta2 = 0.05; BATCHparam.eta0 = 0.9;
    BATCHparam.omega = 0.01; BATCHparam.xi = 0.01; BATCHparam.max_iter = 6;

    GTrain = NormalizeFea(L_train, 1);

    % Hash codes learning
    B = train_BATCH(GTrain, XKTrain, YKTrain, L_train, BATCHparam);

    % Hash functions learning
    XW = (XKTrain' * XKTrain + BATCHparam.xi * eye(size(XKTrain, 2))) \ (XKTrain' * B);
    YW = (YKTrain' * YKTrain + BATCHparam.xi * eye(size(YKTrain, 2))) \ (YKTrain' * B);
    [XKTrain, XKTest, XKRetrieval] = Kernel_Feature_eval(train_dataX, query_dataX, retrieval_dataX, XAnchors);
    [YKTrain, YKTest, YKRetrieval] = Kernel_Feature_eval(train_dataT, query_dataT, retrieval_dataT, YAnchors);

    % Cross-Modal Retrieval
    cbTest_vis = compactbit(XKTest * XW > 0);
    cbDb_vis = compactbit(XKRetrieval * XW > 0);
    cbTest_text = compactbit(YKTest * YW > 0);
    cbDb_text = compactbit(YKRetrieval * YW > 0);

    hamm_T2I = hammingDist(cbTest_text, cbDb_vis)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    hamm_I2T = hammingDist(cbTest_vis, cbDb_text)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
