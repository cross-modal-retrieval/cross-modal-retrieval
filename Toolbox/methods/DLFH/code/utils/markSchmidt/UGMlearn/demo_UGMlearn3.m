clear all
close all
randn('state',0);
rand('state',0);
useMex = 1; % Set to 0 to only use Matlab files

%% Generate a synthetic data set X
fprintf('Generating Synthetic data...\n');
nTrain = 5000;
nTest = 5000;
nNodes = 14;
nStates = 2;
edgeProb = .3;
param = 'I';
[X,edges] = LLM2_generate(nTrain+nTest,nNodes,nStates,edgeProb,param,useMex);
Xtrain = X(1:nTrain,:);
Xtest = X(nTrain+1:end,:);

%% Compute optimal tree
[edgesTree,adjTree] = UGMlearn_optimalTree(X);

%% Set regularization parameter 
% (normally you would search for a good value)
lambda = 10;
options.lambda = lambda;
options.verbose = 0; % Turn off verbose output of optimizer
options.param = 'I';
testInfer = 'exact';
options.useMex = useMex;

%% Run different approximations with Exact Likelihood

fprintf('Using Exact Likelihood with Optimal Tree...');
options.regType = '2';
options.edges = edgesTree;
options.infer = 'tree';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

options.infer = 'exact';

fprintf('Using Exact Likelihood with L2-regularization...');
options.edges = [];
options.regType = '2';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Exact Likelihood with L1-regularization...');
options.regType = '1';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Run different approximations with Pseudo Likelihood

options.infer = 'pseudo';

fprintf('Using Pseudo Likelihood with L2-regularization...');
options.edges = [];
options.regType = '2';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Pseudo Likelihood with L1-regularization...');
options.regType = '1';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Run different approximations with Mean-Field Approximation

options.infer = 'mean';

fprintf('Using Mean-Field Approximation with L2-regularization...');
options.edges = [];
options.regType = '2';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Mean-Field Approximation with L1-regularization...');
options.regType = '1';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Run different approximations with Loopy Approximation

options.infer = 'loopy';

fprintf('Using Loopy Approximation with L2-regularization...');
options.edges = [];
options.regType = '2';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Loopy Approximation with L1-regularization...');
options.regType = '1';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Run different approximations with TRBP Approximation

options.infer = 'trbp';

fprintf('Using TRBP Approximation with L2-regularization...');
options.edges = [];
options.regType = '2';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using TRBP Approximation with L1-regularization...');
options.regType = '1';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);
