function [] = demo_LEMON(bits, dataname)
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

    addpath(genpath('./utils/'));
    addpath(genpath('./codes/'));

    OURparam.alpha = 10000;
    OURparam.beta = 1000;
    OURparam.theta = 0.001;
    OURparam.gamma = 0.01;
    OURparam.xi = 1000;
    run = 5;

    % X:text    Y:img
    OURparam.nbits = bits;
    GTrain_new = NormalizeFea(L_tr, 1);

    [BB, XW, YW, HH] = train_LEMON0(I_tr, T_tr, L_tr, GTrain_new, OURparam);
    cbTest_img = compactbit(I_te * XW > 0);
    cbTest_text = compactbit(T_te * YW > 0);

    cbDb_img = compactbit(I_db * XW > 0);
    cbDb_text = compactbit(T_db * YW > 0);

    hamm_T2I = hammingDist(cbTest_text, cbDb_img)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    hamm_I2T = hammingDist(cbTest_img, cbDb_text)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
