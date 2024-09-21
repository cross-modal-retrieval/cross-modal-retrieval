clear all
close all
randn('state',0);
rand('state',0);
useMex = 1; % Set to 0 to only use Matlab files

%% Generate a synthetic data set X
fprintf('Generating Synthetic data...\n');
nTrain = 2000;
nTest = 2000;
nNodes = 10;
nStates = 3;
edgeProb = .4;
param = 'F';
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
options.infer = 'exact';
testInfer = 'exact';
options.useMex = useMex;

%% Run different methods with Ising parameterization

options.param = 'I';

fprintf('Using Ising parameterization with Optimal Tree...');
options.regType = '2';
options.edges = edgesTree;
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Ising parameterization with L2-regularization...');
options.edges = [];
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using Ising parameterization with L1-regularization...');
options.regType = '1';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Run different methods with gIsing parameterization

options.param = 'P';

fprintf('Using gIsing parameterization with Optimal Tree...');
options.regType = '2';
options.edges = edgesTree;
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using gIsing parameterization with L2-regularization...');
options.edges = [];
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using gIsing parameterization with L1-regularization...');
options.regType = '1';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using gIsing parameterization with Group L1-regularization (L2-norm of groups)...');
options.regType = 'G';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using gIsing parameterization with Group L1-regularization (Linf-norm of groups)...');
options.regType = 'I';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

%% Run different methods with Full parameterization

options.param = 'F';

fprintf('Using full parameterization with Optimal Tree...');
options.regType = '2';
options.edges = edgesTree;
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using full parameterization with L2-regularization...');
options.edges = [];
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using full parameterization with L1-regularization...');
options.regType = '1';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using full parameterization with Group L1-regularization (L2-norm of groups)...');
options.regType = 'G';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using full parameterization with Group L1-regularization (Linf-norm of groups)...');
options.regType = 'I';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);

fprintf('Using full parameterization with Group L1-regularization (Nuclear-norm of groups)...');
options.regType = 'N';
model = LLM2_train(Xtrain,options);
testNLL = model.nll(model,Xtest,testInfer);
fprintf('test NLL = %f\n',testNLL);