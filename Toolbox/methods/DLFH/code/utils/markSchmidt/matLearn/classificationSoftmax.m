function [model] = classificationSoftmax(X,y,options)
% w = classificationSoftmax(X,y,k)
%
% k: number of classes
%
% Computes Maximum Likelihood Softmax Classification parameters

[n,p] = size(X);

if nargin < 3
    options = [];
end

[verbose,lambdaL2,lambdaL1,link] = myProcessOptions(options,'verbose',1,'lambdaL2',0,'lambdaL1',0,'link','softmax');

k = options.nClasses;

if verbose
    optimoptions.Display = 'iter';
else
    optimoptions.Display = 'none';
end

if strcmp(link,'ssvm')
    gradFunc = @SSVMMultiLoss; % One slack per instance/class combination
    w = randn(p,k);
elseif strcmp(link,'ssvm2')
    gradFunc = @SSVMMultiLoss2; % One slack per instance
    w = randn(p,k);
else
    gradFunc = @SoftmaxLoss2;
    w = zeros(p,k-1);
end
if lambdaL2 == 0 && lambdaL1 == 0
    % Maximum Likelihood
    w(:) = minFunc(gradFunc,w(:),optimoptions,X,y,k);
elseif lambdaL1 == 0
    % L2-Regularization
    lambdaVect = lambdaL2*ones(p,k-1);
    lambdaVect(1,:) = 0; % Don't regularize bias elements
    w(:) = minFunc(@penalizedL2,w(:),optimoptions,gradFunc,lambdaVect(:),X,y,k);
elseif lambdaL2 == 0
    % L1-Regularization
    params.verbose = verbose*2;
    lambdaVect = lambdaL1*ones(p,k-1);
    lambdaVect(1,:) = 0; % Don't regularize bias elements
    w(:) = L1GeneralProjection(gradFunc,w(:),lambdaVect(:),params,X,y,k);
else
    params.verbose = verbose*2;
    lambdaVect2 = lambdaL2*ones(p,k-1);
    lambdaVect2(1,:) = 0; % Don't regularize bias elements
    lambdaVect1 = lambdaL1*ones(p,k-1);
    lambdaVect1(1,:) = 0; % Don't regularize bias elements
    w(:) = L1GeneralProjection(@penalizedL2,w(:),lambdaVect1(:),params,gradFunc,lambdaVect2(:),X,y,k);
end


model.nClasses = k;
model.weights = w;
if strcmp(link,'ssvm')
    model.predictFunc = @predictMultiSVM;
else
model.predictFunc = @predictSoftmax;
model.nllFunc = @(model,X,y)SoftmaxLoss2(model.weights,X,y,model.nClasses);
model.errFunc = @(model,X,y)sum(model.predictFunc(model,X) ~= y);
model.lossFunc = @(model,X,y)SoftmaxLoss2(model.weights,X,y,model.nClasses);
end

end

function y = predictSoftmax(model,X)
k = model.nClasses;
w = model.weights;
[n,p] = size(X);
[junk y] = max(X*[w zeros(p,1)],[],2);
end

function y = predictMultiSVM(model,X)
k = model.nClasses;
w = model.weights;
[n,p] = size(X);
[junk y] = max(X*w,[],2);
end

function lik = likSoftmax(model,X,y)
% Note: untested function
k = model.nClasses;
w = model.weights;
[n,p] = size(X);
lik = exp(X*[w zeros(p,1)]);
lik = normalizeRows(lik);
end
