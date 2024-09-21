function [model] = classificationKernelSoftmax(X,y,options)
% w = classificationKernelSoftmax(X,y)

[n,p] = size(X);

if nargin < 3
    options = [];
end

[verbose,kernelFunc,kernelArgs,lambdaL2,link] = myProcessOptions(options,'verbose',1,'kernelFunc',@kernelLinear,'kernelArgs',{},'lambdaL2',1e-5,'link','softmax');

if verbose
    optimoptions.Display = 'iter';
else
    optimoptions.Display = 'none';
end

k = options.nClasses;
K = kernelFunc(X,X,kernelArgs);
gradArgs = {K,y,k};

% L2-Regularization
if strcmp(link,'ssvm')
    fprintf('SVM!');
    lossFunc = @SSVMMultiLoss;
    u = zeros(n,k);
    u(:) = minFunc(@penalizedKernelL2_matrix,u(:),optimoptions,K,k,lossFunc,lambdaL2,gradArgs{:});
else
    lossFunc = @SoftmaxLoss2;
    u = zeros(n,k-1);
    u(:) = minFunc(@penalizedKernelL2_matrix,u(:),optimoptions,K,k-1,lossFunc,lambdaL2,gradArgs{:});
end

model.nTrain = n;
model.nClasses = k;
model.weights = u;
model.Xtrain = X;
model.kernelFunc = kernelFunc;
model.kernelArgs = kernelArgs;
if strcmp(link,'ssvm')
    fprintf('SVM!');
    model.predictFunc = @predictSVM;
else
    model.predictFunc = @predictSoftmax;
end
model.lossFunc = @(model,X,y)SoftmaxLoss2(model.weights,model.kernelFunc(X,model.Xtrain,kernelArgs),y,model.nClasses);

end

function y = predictSoftmax(model,X)
k = model.nClasses;
u = model.weights;
X = model.kernelFunc(X,model.Xtrain,model.kernelArgs);
[n,p] = size(X);
[junk y] = max(X*[u zeros(model.nTrain,1)],[],2);
end

function y = predictSVM(model,X)
k = model.nClasses;
u = model.weights;
X = model.kernelFunc(X,model.Xtrain,model.kernelArgs);
[n,p] = size(X);
[junk y] = max(X*u,[],2);
end