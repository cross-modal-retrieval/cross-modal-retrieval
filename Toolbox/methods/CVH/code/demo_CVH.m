function [] = demo_CVH(bits, dataname)

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

    k_near = 5;
    bits = str2num(bits);

    conf.randSeed = 1;
    randn('state', conf.randSeed);
    rand('state', conf.randSeed);

    feaTrain_text = T_tr;
    clear T_tr;
    feaTrain_vis = I_tr;
    clear I_tr;
    feaTrain_class = L_tr;
    clear L_tr;

    feaTest_text = T_te;
    clear T_te;
    feaTest_vis = I_te;
    clear I_te;
    feaTest_class = L_te;

    feaDb_text = T_db;
    clear T_db;
    feaDb_vis = I_db;
    clear I_db;
    feaDb_class = L_db;

    CVHparam.lambda = 1e-4;

    [model, drop] = CVH_learn(feaTrain_vis, feaTrain_text, CVHparam, bits);

    codeDb_T = CVH_compress1(feaDb_text, model);
    codeDb_I = CVH_compress(feaDb_vis, model);

    codeTest_T = CVH_compress1(feaTest_text, model);
    codeTest_I = CVH_compress(feaTest_vis, model);

    %% evaluation
    codeDb_T = codeDb_T';
    codeDb_I = codeDb_I';

    codeTest_T = codeTest_T';
    codeTest_I = codeTest_I';

    cbDb_T = compactbit(codeDb_T(:, 1:bits));
    cbDb_I = compactbit(codeDb_I(:, 1:bits));

    cbTest_T = compactbit(codeTest_T(:, 1:bits));
    cbTest_I = compactbit(codeTest_I(:, 1:bits));

    hamm_T2I = hammingDist(cbTest_T, cbDb_I)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    hamm_I2T = hammingDist(cbTest_I, cbDb_T)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
