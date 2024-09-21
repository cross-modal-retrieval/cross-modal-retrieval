function [MAP_I2T, MAP_T2I] = DLFH_algo(dataset, param)
    trainL = dataset.databaseL;

    [BX_opt, BY_opt] = DLFH(trainL, param);

    XTrain = dataset.XDatabase;
    YTrain = dataset.YDatabase;

    Wx = (XTrain' * XTrain + param.gamma * eye(size(XTrain, 2))) \ ...
        XTrain' * BX_opt;
    Wy = (YTrain' * YTrain + param.gamma * eye(size(YTrain, 2))) \ ...
        YTrain' * BY_opt;

    XTest = dataset.XTest;
    YTest = dataset.YTest;
    learn_dataX = dataset.learn_dataX;
    learn_dataT = dataset.learn_dataT;
    L_te = dataset.L_te;
    L_db = dataset.L_db;

    tic
    B_tstx_code = XTest * Wx;
    B_tstt_code = YTest * Wy;
    B_trnx_code = learn_dataX * Wx;
    B_trnt_code = learn_dataT * Wy;

    cbTest_vis = compactbit(B_tstx_code > 0);
    cbTest_text = compactbit(B_tstt_code > 0);
    cbDb_vis = compactbit(B_trnx_code > 0);
    cbDb_text = compactbit(B_trnt_code > 0);

    hamm_T2I = hammingDist(cbTest_text, cbDb_vis)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    hamm_I2T = hammingDist(cbTest_vis, cbDb_text)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);
end
