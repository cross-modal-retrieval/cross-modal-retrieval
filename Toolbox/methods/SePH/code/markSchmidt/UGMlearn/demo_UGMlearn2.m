clear all
close all
randn('state',0);
rand('state',0);
useMex = 1; % Set to 0 to only use Matlab files

%% Generate a synthetic data set X
fprintf('Generating Synthetic data...\n');
nTrain = 10000;
nTest = 10000;
nNodes = 10;
nStates = 2;
edgeProb = .4;
param = 'F';
[X,edges] = LLM2_generate(nTrain+nTest,nNodes,nStates,edgeProb,param,useMex);
Xtrain = X(1:nTrain,:);
Xtest = X(nTrain+1:end,:);

%% Set sequence of regularization parameters

lambdaValues = 2.^[10:-.25:-8];

%% Set up model parameters
%options.verbose = 0; % Turn off verbose output of optimizer
options.regType = '1';
options.param = 'F';
options.infer = 'exact';
testInfer = 'exact';
options.useMex = useMex;

%% Train and test w/ sequence of regularization parameters
testNLL = inf(length(lambdaValues),1);
minNLL = inf;
for regParam = 1:length(lambdaValues);
    options.lambda = lambdaValues(regParam);
    if regParam == 1
        model = LLM2_trainActive(Xtrain,options);
    else
        model = LLM2_trainActive(Xtrain,options,model);
    end
    testNLL(regParam) = model.nll(model,Xtest,testInfer);
    fprintf('lambda = %f, testNLL = %f, nnz = %d\n',lambdaValues(regParam),testNLL(regParam),nnz(model.w));
    
    if testNLL(regParam) < minNLL
        minLambda = lambdaValues(regParam);
        minNLL = testNLL(regParam);
        minModel = model;
    end
    
    if regParam > max(6,length(lambdaValues)/4) && issorted(testNLL(regParam-5:regParam))
        fprintf('Test NLL increased on 5 consecutive iterations, terminating.\n');
        break;
    end
end
fprintf('Best lambda = %f, nnz = %d\n',minLambda,nnz(minModel.w));
