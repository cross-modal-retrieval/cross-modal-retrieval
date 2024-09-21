%% Load data
clear all

fprintf('Loading Data\n');
load('rcv1_train.binary.mat');
X = [ones(size(X,1),1) X];
[n,p] = size(X);

%% Set up problem
maxIter = n*20; % 20 passes through the data set
lambda = 1/n;

objective = @(w)(1/n)*LogisticLoss(w,X,y) + (lambda/2)*(w'*w);

% Order of examples to process
iVals = int32(ceil(n*rand(maxIter,1)));

%%
fprintf('Running Pegasos-style SG method with 1/(k*lambda) step size and projection step\n');

Xt = X'; % Function works with transpose of X
stepSizes = (1/lambda)./[1:maxIter]';
maxNorm = sqrt(2*log(2)/lambda);

w = zeros(p,1);
SGD_logistic(w,Xt,y,lambda,stepSizes,iVals,maxNorm); 
% Note that w is updated in place

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running basic SG method with constant step size\n');

stepSizes = 1e-1*ones(maxIter,1);

w = zeros(p,1);
SGD_logistic(w,Xt,y,lambda,stepSizes,iVals);

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running averaged SG method with a constant step size\n');

stepSizes = 1*ones(maxIter,1);

w = zeros(p,1);
wAvg = ASGD_logistic(w,Xt,y,lambda,stepSizes,iVals);
% Note that ASGD_logistic is faster than SGD_logistic for sparse
% data when computing the uniform averaging

f = objective(wAvg);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic primal coordinate descent\n');
maxIterPCD = round(maxIter*p/n); % Number of PCD iterations to do roughly the same amount of work
yX = diag(sparse(y))*X;
Lj = (.25/n)*sum(X.^2)' + lambda; % Lipschitz constants of each coordinate

% Order of variables to process
jVals = int32(ceil(p*rand(maxIterPCD,1)));

w = zeros(p,1);
PCD_logistic(w,yX,lambda,Lj,jVals);
% Note that you can pass in yXw = yX*w as an additional argument to avoid
% this computation. The vector will be updated in-place.

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic dual coordinate ascent\n');
yXt = yX';

alpha = 1e-10*ones(n,1);
DCA_logistic(alpha,yXt,lambda,iVals);
% Note that you can pass w = -yXt*alpha/(lambda*nSamples) and yxxy = sum(yXt.^2)'
% as additional arguments to avoid this computation. 
% The vector w will be updated in place
w = yXt*alpha/(lambda*n);

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic average gradient with constant step size\n');

stepSize = 1;
d = zeros(p,1);
g = zeros(n,1);
covered = int32(zeros(n,1));

w = zeros(p,1);
SAG_logistic(w,Xt,y,lambda,stepSize,iVals,d,g,covered);
% {w,d,g,covered} are updated in-place

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic average gradient with line-search\n');

d = zeros(p,1);
g = zeros(n,1);
covered = int32(zeros(n,1));
Lmax = 1;

w = zeros(p,1);
SAGlineSearch_logistic(w,Xt,y,lambda,Lmax,iVals,d,g,covered);
% Lmax is also updated in-place
% You pass another argument set to int32(2) to use 2/(Lmax+n*mu) instead of
% 1/Lmax
% You can further pass in xtx = sum(X.^2,2) to avoid this computation

f = objective(w);
fprintf('f = %.6f\n',f);
