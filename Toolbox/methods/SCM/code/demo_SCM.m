function [] = demo_SCM(bits, dataname)
    lambda = 1e-6;
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

    % X:text  Y:image
    XTrain = T_tr;
    YTrain = I_tr;
    clear T_tr I_tr;

    XTest = T_te;
    YTest = I_te;
    clear T_te I_te;

    XDb = T_db;
    YDb = I_db;
    clear T_db I_db;

    dimX = size(XTrain, 2);
    dimY = size(YTrain, 2);

    XMean = mean(XTrain);
    XTrain = bsxfun(@minus, XTrain, XMean);
    XTest = bsxfun(@minus, XTest, XMean);

    YMean = mean(YTrain);
    YTrain = bsxfun(@minus, YTrain, YMean);
    YTest = bsxfun(@minus, YTest, YMean);

    Cxx = XTrain' * XTrain + lambda * eye(dimX);
    Cyy = YTrain' * YTrain + lambda * eye(dimY);
    Cxy = XTrain' * YTrain;

    % evaluation
    WTrue = (L_te * L_db' >= 1);

    [SCMsWx, SCMsWy] = trainSCM_seq(XTrain, YTrain, bits, L_tr);

    BxTest = compactbit(XTest * SCMsWx >= 0);
    ByDb = compactbit(YDb * SCMsWy >= 0);
    hamm_T2I = hammingDist(BxTest, ByDb)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    ByTest = compactbit(YTest * SCMsWy >= 0);
    BxDb = compactbit(XDb * SCMsWx >= 0);
    hamm_I2T = hammingDist(ByTest, BxDb)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end

function cb = compactbit(b)
    %
    % b = bits array
    % cb = compacted string of bits (using words of 'word' bits)
    [nSamples nbits] = size(b);
    nwords = ceil(nbits / 8);
    cb = zeros([nSamples nwords], 'uint8');

    for j = 1:nbits
        w = ceil(j / 8);
        cb(:, w) = bitset(cb(:, w), mod(j - 1, 8) + 1, b(:, j));
    end

end
