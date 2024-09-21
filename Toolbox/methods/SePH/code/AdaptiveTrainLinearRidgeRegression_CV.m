function [weights, gaussian, regulazor, optMSE] = AdaptiveTrainLinearRidgeRegression_CV( X, Y, nFold)
% Adaptively learn linear ridge regression via n-fold cross-validation,
% X: n x f, feature matrix, already added a column of 1s
% Y: n x c, label matrix
% nFold: number of $n$ for n-fold cross-validation

n = size(X,1);

%Index postive examples and negative examples
index = randperm(n);
trainFold = cell(1, nFold);
testFold = cell(1, nFold);

for i = 1 : nFold
    trainFold{i} = index( [1 : round((i-1)*n/nFold), round((i)*n/nFold)+1 : n] );
    testFold{i} = index( round((i-1)*n/nFold)+1 : round((i)*n/nFold) );
end

%Find the best mu
mu = 10 .^ [-3:3]; 

% Change the iteration for acceleration
errors = zeros(length(mu), size(Y, 2));
validateCandidate = cell(nFold, length(mu));

for j = 1 : nFold    
    for i = 1 : length(mu)
        tmpY = Y(trainFold{j},:);
        tmpX = X(trainFold{j},:);       
        validateCandidate{j, i} = (tmpX' * tmpX + mu(i) * eye(size(tmpX, 2)))^-1 * tmpX';
        weights = validateCandidate{j, i} * tmpY;
        testError = mean( ( X(testFold{j},:) * weights - Y(testFold{j},:) ).^2 ); 
        errors(i, :) = errors(i, :) + testError;
    end
end
errors = errors / nFold;

testCandidate = cell(1, length(mu));
for i = 1 : length(mu)
    tmpX = X; 
    testCandidate{i} = (tmpX' * tmpX + mu(i) * eye(size(tmpX, 2)))^-1 * tmpX';
end

nLabel = size(Y, 2);
optMSE = zeros(1, nLabel);
regulazor = zeros(1, nLabel);
weights = zeros(size(X, 2), nLabel);

gaussian = zeros(2, 2, nLabel);
for i = 1 : nLabel
    tmperror = errors(:, i);
    C = min(tmperror);
    C2 = min(tmperror(length(tmperror):-1:1));
    Is = find(tmperror==C);
    I = Is(round(length(Is)/2));
    if C~=C2
        C
        C2
        error('C cannot be different to C2');
    end
    
    optMSE(1, i) = C;
    regulazor(1, i) = mu(I);    
    weights(:, i) = testCandidate{I} * Y(:, i);
    
    % Estimate the mu / sigma in Gaussian distributions
    v_Y1 = [];
    g_Y1 = [];
    for j = 1 : nFold    
        tmpY = Y(trainFold{j},i);
        weights0 = validateCandidate{j, I} * tmpY;
        v_Y = X(testFold{j}, :) * weights0;
        g_Y = Y(testFold{j}, i);
        v_Y1 = [v_Y1; v_Y];
        g_Y1 = [g_Y1; g_Y];
    end
    
    posV = v_Y1(g_Y1 == 1);
    negV = v_Y1(g_Y1 == -1);
    
    if length(posV) > 0
        gaussian(1, 1, i) = mean(posV);
        gaussian(2, 1, i) = std(posV, 1);
    end
    
    if length(negV) > 0
        gaussian(1, 2, i) = mean(negV);
        gaussian(2, 2, i) = std(negV, 1);
    end
end        