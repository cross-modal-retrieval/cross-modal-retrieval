function [] = runDCH_discrete(bits, dataname)
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

    nanchor = 1000;
    I_mean = mean(I_tr, 1);
    T_mean = mean(T_tr, 1);
    I_tr = bsxfun(@minus, I_tr, I_mean);
    T_tr = bsxfun(@minus, T_tr, T_mean);
    [model, B] = DCH_train(I_tr, T_tr, I_te, T_te, L_tr, L_te, nanchor, bits);

    I_te = bsxfun(@minus, I_te, I_mean);
    T_te = bsxfun(@minus, T_te, T_mean);

    I_db = bsxfun(@minus, I_db, I_mean);
    T_db = bsxfun(@minus, T_db, T_mean);

    B_I_te = sign(I_te * model.W);
    B_T_te = sign(T_te * model.Wt);
    %B_db = sign(B);
    B_I_db = sign(I_db * model.W);
    B_T_db = sign(T_db * model.Wt);

    CB_I_te = compactbit(B_I_te > 0);
    CB_T_te = compactbit(B_T_te > 0);
    CB_I_db = compactbit(B_I_db > 0);
    CB_T_db = compactbit(B_T_db > 0);

    hamm_I2T = hammingDist(CB_I_te, CB_T_db)';
    MAP_I2T = perf_metric4Label(L_db, L_te, hamm_I2T);

    hamm_T2I = hammingDist(CB_T_te, CB_I_db)';
    MAP_T2I = perf_metric4Label(L_db, L_te, hamm_T2I);

    name = ['../result/' dataname '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%d] MAP@I2T = %.4f, MAP@T2I = %.4f\n', dataname, bits, MAP_I2T, MAP_T2I);
    fclose(fid);

end

%%
%% function SDH_train use the SDL method for two modalities in the same framework
%  for X and T, we use different anchor
function [F, B] = DCH_train(train_abstract_X_opt, train_abstract_T_opt, test_abstract_X_opt, test_abstract_T_opt, Ltraining, Ltest, n_anchors, nbits)

    traindata_X = double(train_abstract_X_opt);
    traindata_T = double(train_abstract_T_opt);
    testdata_X = double(test_abstract_X_opt);
    testdata_T = double(test_abstract_T_opt);

    traingnd = single(Ltraining);
    testgnd = single(Ltest);

    Ntrain = size(traindata_X, 1);
    % Use all the training data
    X = traindata_X;
    T = traindata_T;
    label = double(traingnd);

    % get anchors
    % n_anchors = Ntrain;
    % n_anchors = 1000;
    % rand('seed',1);

    if n_anchors == 0
        n_anchors = Ntrain;
    end

    anchor = X(randsample(Ntrain, n_anchors), :); %û��
    anchor2 = T(randsample(Ntrain, n_anchors), :); %û��

    % % determin rbf width sigma
    % Dis = EuDist2(X,anchor,0);
    % % sigma = mean(mean(Dis)).^0.5;
    % sigma = mean(min(Dis,[],2).^0.5);
    % clear Dis
    sigma = 0.4; % for normalized data   û��

    PhiX = X;
    PhiT = T;
    Phi_testdata_X = testdata_X;
    Phi_testdata_T = testdata_T;
    Phi_traindata_X = traindata_X;
    Phi_traindata_T = traindata_T;

    % learn G and F
    maxItr = 5;
    gmap.lambda = 1; gmap.loss = 'L2';
    Fmap.type = 'RBF';
    Fmap.nu = 0; %  penalty parm for F term
    Fmap.mu = 0;
    Fmap.lambda = 1e-2;

    %% run algo
    % nbits = 32;

    % Init Z
    % randn('seed',3);
    Zinit = sign(randn(Ntrain, nbits));

    debug = 0;
    [~, F, B] = DCH_discrete(PhiX, PhiT, label, Zinit, gmap, Fmap, [], maxItr, debug);
    % [~, F, H] = SDH_NODES(PhiX,PhiT, label,Zinit,gmap,Fmap,[],maxItr,debug);

    %% evaluation
    % display(sprintf('Evaluation with %d bits, %d achors for rfb... \n', nbits, n_anchors));

    %AsymDist = 0; % Use asymmetric hashing or not
    %
    %if AsymDist
    %    H = H > 0; % directly use the learned bits for training data
    %else
    %    HX = Phi_traindata_X*F.W;
    %    HT = Phi_traindata_T*F.Wt;
    %
    %    H = H;
    %end

    %tHX = Phi_testdata_X*F.W;
    %tHT = Phi_testdata_T*F.Wt;
    %
    %
    %BX = sign(HX);
    %BT = sign(HT);
    %tBX = sign(tHX);
    %tBT = sign(tHT);
    %B = sign(H);
    %
    %%% use another evaluation metrics
    %BX = compactbit(sign(HX) >= 0);
    %BT = compactbit(sign(HT) >= 0);
    %tBX = compactbit(sign(tHX) >= 0);
    %tBT = compactbit(sign(tHT) >= 0);
    %B = compactbit(sign(H) >= 0);
    %
    %hammingM = hammingDist(tBX, B)';
    %[ MAP_I2T ] = perf_metric4Label( traingnd, testgnd, hammingM );
    %% fprintf('... image to text: %.4f\n', MAP_I2T);
    %
    %hammingM = hammingDist(tBT, B)';
    %[ MAP_T2I ] = perf_metric4Label( traingnd, testgnd, hammingM );
    % fprintf('... text to image: %.4f\n', MAP_T2I);

end
