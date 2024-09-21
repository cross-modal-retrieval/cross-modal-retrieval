%% Load data
clear all
fprintf('Loading Data\n');
load('covtype.libsvm.binary.mat');
X = [ones(size(X,1),1) standardizeCols(X)];
y(y==2) = -1;
[n,p] = size(X);

rand('state',0);
randn('state',0);

%% Set up problem
maxIter = n*10; % 10 passes through the data set
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
SGD_logistic_BLAS(w,Xt,y,lambda,stepSizes,iVals,maxNorm); 
% Note that w is updated in place

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running averaged Pegsos-style SG method with 1/(k*lambda)\n');

stepSizes = (1/lambda)./[1:maxIter]';
maxNorm = sqrt(2*log(2)/lambda);

w = zeros(p,1);
wAvg = SGD_logistic_BLAS(w,Xt,y,lambda,stepSizes,iVals,maxNorm); 
% Note that averaged is computed and returned if there is a return argument

f = objective(wAvg);
fprintf('f = %.6f\n',f);

%%
fprintf('Running averaged SG method with 1/(k*lambda) with tail-weighted averaging\n');

stepSizes = (1/lambda)./[1:maxIter]';
maxNorm = sqrt(2*log(2)/lambda);
averageWeights = [1:maxIter]'/sum(1:maxIter);

% Use this to only average the second half of the iterations:
%averageWeights = [zeros(maxIter/2,1);ones(maxIter/2,1)/(maxIter/2)];

w = zeros(p,1);
wAvg = SGD_logistic_BLAS(w,Xt,y,lambda,stepSizes,iVals,maxNorm,averageWeights);

f = objective(wAvg);
fprintf('f = %.6f\n',f);

%%
fprintf('Running basic SG method with constant step size\n');

stepSizes = 1e-4*ones(maxIter,1);

w = zeros(p,1);
SGD_logistic_BLAS(w,Xt,y,lambda,stepSizes,iVals);

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running averaged SG method with a constant step size\n');

stepSizes = 1e-3*ones(maxIter,1);

w = zeros(p,1);
wAvg = ASGD_logistic_BLAS(w,Xt,y,lambda,stepSizes,iVals);
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
yXw = yX*w;
PCD_logistic_BLAS(w,yX,lambda,Lj,jVals,yXw);
% Note that yXw is updated in place

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running primal coordinate descent with Lipschitz sampling\n');

% Generate samples according to Lipschitz constants of variables
cs = cumsum(Lj)/sum(Lj);
jVals_Lipschitz = sampleDiscrete_cumsumC(cs,rand(maxIterPCD,1));

w = zeros(p,1);
yXw = yX*w;
PCD_logistic_BLAS(w,yX,lambda,Lj,jVals_Lipschitz,yXw);

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic dual coordinate ascent\n');
yXt = yX';
yxxy = sum(yXt.^2)';

alpha = 1e-10*ones(n,1);
w = -yXt*alpha/(lambda*n);
DCA_logistic_BLAS(alpha,yXt,lambda,iVals,w,yxxy);
% Note that w is updated in place, and it returns the negative of w

f = objective(-w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic average gradient with constant step size\n');

stepSize = 1e-3;
d = zeros(p,1);
g = zeros(n,1);
covered = int32(zeros(n,1));

w = zeros(p,1);
SAG_logistic_BLAS(w,Xt,y,lambda,stepSize,iVals,d,g,covered);
% {w,d,g,covered} are updated in-place

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic average gradient with line-search\n');

xtx = sum(X.^2,2);

d = zeros(p,1);
g = zeros(n,1);
covered = int32(zeros(n,1));
Lmax = 1;

w = zeros(p,1);
SAGlineSearch_logistic_BLAS(w,Xt,y,lambda,Lmax,iVals,d,g,covered,int32(1),xtx);
% Lmax is also updated in-place

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic average gradient with constant step size and Lipschitz sampling\n');

Li = (.25/n)*sum(X.^2,2) + lambda; % Lipschitz constants of each example

% Generate samples according to Lipschitz constants of example
cs = cumsum(Li)/sum(Li);
iVals_Lipschitz = sampleDiscrete_cumsumC(cs,rand(maxIter,1));

stepSize = 1e-3;
d = zeros(p,1);
g = zeros(n,1);
covered = int32(zeros(n,1));

w = zeros(p,1);
SAG_logistic_BLAS(w,Xt,y,lambda,stepSize,int32(iVals_Lipschitz),d,g,covered);

f = objective(w);
fprintf('f = %.6f\n',f);

%%
fprintf('Running stochastic average gradient with line-search and adaptive Lipschitz sampling\n');

Lmax = 1; % Initial guess of overall Lipschitz constant
Li = ones(n,1); % Initial guess of Lipschitz constant of each function

randVals = rand(maxIter,2); % Random values to be used by the algorithm

d = zeros(p,1);
g = zeros(n,1);
covered = int32(zeros(n,1));

w = zeros(p,1);
SAG_LipschitzLS_logistic(w,Xt,y,lambda,Lmax,Li,randVals,d,g,covered,int32(1),xtx);
% Li is also updated in-place

f = objective(w);
fprintf('f = %.6f\n',f);