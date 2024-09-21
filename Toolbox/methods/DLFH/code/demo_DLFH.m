function [] = demo_DLFH(bits, dataname)
    addpath(genpath(fullfile('utils/')));
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

    param.bits = bits;
    param.num_samples = param.bits;
    param.maxIter = 30;
    param.gamma = 1e-2;
    param.lambda = 8;
    % param.dataset_name = dataset_name;

    dataset.XDatabase = I_tr;
    dataset.YDatabase = T_tr;
    dataset.XTest = I_te;
    dataset.YTest = T_te;
    dataset.learn_dataX = I_db;
    dataset.learn_dataT = T_db;
    dataset.databaseL = L_tr;
    dataset.L_te = L_te;
    dataset.L_db = L_db;

    [MAP_I2T, MAP_T2I] = DLFH_algo(dataset, param);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
