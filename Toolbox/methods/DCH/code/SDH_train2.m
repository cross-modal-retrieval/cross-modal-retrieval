%% function SDH_train use the SDL method for two modalities in the same framework
%  for X and T, we use different anchor
function [F, H, MAP_I2T, MAP_T2I] = SDH_train2(train_abstract_X_opt, train_abstract_T_opt, test_abstract_X_opt, test_abstract_T_opt, Ltraining, Ltest, n_anchors, nbits, islinear)


traindata_X = double(train_abstract_X_opt');
traindata_T = double(train_abstract_T_opt');
testdata_X = double(test_abstract_X_opt');
testdata_T = double(test_abstract_T_opt');

traingnd = single(Ltraining);
testgnd = single(Ltest);


Ntrain = size(traindata_X,1);
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

anchor = X(randsample(Ntrain, n_anchors),:);
anchor2 = T(randsample(Ntrain, n_anchors),:);

% % determin rbf width sigma
% Dis = EuDist2(X,anchor,0);
% % sigma = mean(mean(Dis)).^0.5;
% sigma = mean(min(Dis,[],2).^0.5);
% clear Dis
sigma = 0.4; % for normalized data

if islinear
    PhiX = X;
    PhiT = T;
    Phi_testdata_X = testdata_X;
    Phi_testdata_T = testdata_T;
    Phi_traindata_X = traindata_X;
    Phi_traindata_T = traindata_T;
else
    PhiX = exp(-sqdist(X,anchor)/(2*sigma*sigma));
    PhiX = [PhiX, ones(Ntrain,1)];

    PhiT = exp(-sqdist(T,anchor2)/(2*sigma*sigma));
    PhiT = [PhiT, ones(Ntrain,1)];

    Phi_testdata_X = exp(-sqdist(testdata_X,anchor)/(2*sigma*sigma)); 
    Phi_testdata_X = [Phi_testdata_X, ones(size(Phi_testdata_X,1),1)];

    Phi_testdata_T = exp(-sqdist(testdata_T,anchor2)/(2*sigma*sigma)); 
    Phi_testdata_T = [Phi_testdata_T, ones(size(Phi_testdata_T,1),1)];

    Phi_traindata_X = exp(-sqdist(traindata_X,anchor)/(2*sigma*sigma)); 
    Phi_traindata_X = [Phi_traindata_X, ones(size(Phi_traindata_X,1),1)];

    Phi_traindata_T = exp(-sqdist(traindata_T,anchor2)/(2*sigma*sigma)); 
    Phi_traindata_T = [Phi_traindata_T, ones(size(Phi_traindata_T,1),1)];
end



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
randn('seed',3);
Zinit=sign(randn(Ntrain,nbits));


debug = 0;
% [~, F, H] = SDH_CDL(PhiX,PhiT, label,Zinit,gmap,Fmap,[],maxItr,debug);
[~, F, H] = SDH_NODES(PhiX,PhiT, label,Zinit,gmap,Fmap,[],maxItr,debug);

%% evaluation
% display(sprintf('Evaluation with %d bits, %d achors for rfb... \n', nbits, n_anchors));

AsymDist = 0; % Use asymmetric hashing or not

if AsymDist 
    H = H > 0; % directly use the learned bits for training data
else
    HX = Phi_traindata_X*F.W;
    HT = Phi_traindata_T*F.Wt;
    
    H = H;
end

tHX = Phi_testdata_X*F.W;
tHT = Phi_testdata_T*F.Wt;


BX = sign(HX);
BT = sign(HT);
tBX = sign(tHX);
tBT = sign(tHT);
B = sign(H);


% fprintf('using CMDH method: \n');  

% sim = B * tBX';
% MAP_I2T = mAP(sim,traingnd,testgnd, 0);
% fprintf('... image to text: %.4f\n', MAP_I2T);
% 
% sim = B * tBT';
% MAP_T2I = mAP(sim,traingnd,testgnd, 0);
% fprintf('... text to image: %.4f\r\n', MAP_T2I);

%% use another evaluation metrics
BX = compactbit(sign(HX) >= 0);
BT = compactbit(sign(HT) >= 0);
tBX = compactbit(sign(tHX) >= 0);
tBT = compactbit(sign(tHT) >= 0);
B = compactbit(sign(H) >= 0);

hammingM = hammingDist(tBX, B)';
[ MAP_I2T ] = perf_metric4Label( traingnd, testgnd, hammingM );
% fprintf('... image to text: %.4f\n', MAP_I2T);

hammingM = hammingDist(tBT, B)';
[ MAP_T2I ] = perf_metric4Label( traingnd, testgnd, hammingM );
% fprintf('... text to image: %.4f\n', MAP_T2I);


% fprintf('finished! \n');

