function [] = demo_LSSH(bits, dataname)
    addpath(genpath('./SLEP_package_4.1/SLEP/'));
    data_dir = '../../Data';

    if ~exist(data_dir, 'dir')
        error('No such dir(%s)', fullfile(pwd, data_dir))
    end

    if ~exist('../result', 'dir')
        mkdir('../result')
    end

    addpath(data_dir)

    if strcmp(dataname, 'flickr')
        load('mir_cnn.mat');
    elseif strcmp(dataname, 'nuswide')
        load('nus_cnn.mat');
    elseif strcmp(dataname, 'coco')
        load('coco_cnn.mat');
    else
        fprintf('ERROR dataname!');
    end

    addpath(data_dir);
    bits = str2num(bits);
    mu = 0.05;
    rho = 0.5;
    lambda = 0.2;

    I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
    I_db = bsxfun(@minus, I_db, mean(I_tr, 1));
    I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));

    T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
    T_db = bsxfun(@minus, T_db, mean(T_tr, 1));
    T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

    %     % construct training set
    I_temp = I_tr';
    T_temp = T_tr';
    [row, col] = size(I_temp);
    [rowt, colt] = size(T_temp);

    I_temp = bsxfun(@minus, I_temp, mean(I_temp, 2));
    T_temp = bsxfun(@minus, T_temp, mean(T_temp, 2));
    Im_te = (bsxfun(@minus, I_te', mean(I_tr', 2)))';
    Te_te = (bsxfun(@minus, T_te', mean(T_tr', 2)))';

    %addpath('../common/');

    opts = [];
    opts.mu = mu;
    opts.rho = rho;
    opts.lambda = lambda;
    opts.maxOutIter = 20;

    [B, PX, PT, R, A, S, opts] = solveLSSH(I_temp', T_temp', bits, opts); %����LSSH��������

    [train_hash, test_hash_I, test_hash_T] = LSSHcoding(A, B, PX, PT, R, I_db, T_db, Im_te, Te_te, opts, bits);
    train_hash = train_hash > 0;
    train_hash = train_hash';
    cbDb_comm = compactbit(train_hash(:, 1:bits));

    test_hash_T = test_hash_T > 0;
    test_hash_T = test_hash_T';
    cbTest_text = compactbit(test_hash_T(:, 1:bits));

    test_hash_I = test_hash_I > 0;
    test_hash_I = test_hash_I';
    cbTest_img = compactbit(test_hash_I(:, 1:bits));

    hamm_T2I = hammingDist(cbTest_text, cbDb_comm)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    hamm_I2T = hammingDist(cbTest_img, cbDb_comm)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
