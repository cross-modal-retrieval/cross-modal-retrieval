function [] = demo_IMH(bits, dataname)
    addpath('toolbox');
    vl_setup;
    data_dir = '../../Data';

    if ~exist(data_dir, 'dir')
        error('No such dir(%s)', fullfile(pwd, data_dir))
    end

    addpath(data_dir);

    if ~exist('../result', 'dir')
        mkdir('../result')
    end

    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    bits = str2num(bits);

    conf.randSeed = 1;
    randn('state', conf.randSeed);
    rand('state', conf.randSeed);
    vl_twister('state', conf.randSeed);

    top_num = 1;
    param.beta = 1;
    param.lambda = 1;
    param.k = 1;
    param.sigma = 1;

    [model, drop] = IMH_learn(I_tr, T_tr, param, bits);

    % T2I ================================================================
    codeTest_text = IMH_compress1(T_te, model);
    codeDb_vis = IMH_compress(I_db, model);

    cbDb_vis = compactbit(codeDb_vis(:, 1:bits));
    cbTest_text = compactbit(codeTest_text(:, 1:bits));

    hamm_T2I = hammingDist(cbTest_text, cbDb_vis)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    % I2T ================================================================
    codeTest_vis = IMH_compress(I_te, model);
    codeDb_text = IMH_compress1(T_db, model);

    cbTest_vis = compactbit(codeTest_vis(:, 1:bits));
    cbDb_text = compactbit(codeDb_text(:, 1:bits));

    hamm_I2T = hammingDist(cbTest_vis, cbDb_text)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
