function [] = demo_STMH(bits, dataname)
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

    bits = str2num(bits);

    % input data
    inXCell = cell(2, 1);
    inXCell{1, 1} = I_tr';
    inXCell{2, 1} = T_tr';
    % parameters
    inPara.maxIter = 10;
    inPara.thresh = 0.01;
    rr = 0.5;
    inPara.r = [rr, 1 - rr];
    inPara.lamda = 0.5;
    inPara.gamma = 0.0001;

    inPara.bits = bits;

    numzeros = round(inPara.bits / 2);

    %[G,outFCell,R] = multi_STMH(inXCell, inPara, numzeros);
    [G, outFCell, R] = multi_STMH(I_tr', T_tr', inPara, numzeros);
    B_T_te = compress_text(T_te, outFCell, numzeros);
    B_T_db = compress_text(T_db, outFCell, numzeros);

    B_I_te = compress_img(I_te, outFCell, R, inPara);
    B_I_db = compress_img(I_db, outFCell, R, inPara);

    % T2I ================================================================
    hamm_T2I = hammingDist(B_T_te, B_I_db)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    % I2T ================================================================
    hamm_I2T = hammingDist(B_I_te, B_T_db)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end
