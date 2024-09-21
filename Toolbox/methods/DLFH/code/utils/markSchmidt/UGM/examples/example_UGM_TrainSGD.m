%% Load Noisy X
clear all
close all

load X.mat
[nRows,nCols] = size(X);
nNodes = nRows*nCols;
nStates = 2;
nInstances = 100;

% Make 100 noisy X instances
y = int32(1+X);
y = reshape(y,[1 1 nNodes]);
y = repmat(y,[nInstances 1 1]);

X = reshape(X,1,1,nNodes);
X = repmat(X,[nInstances 1 1]);
X = X + randn(size(y))/2;

figure;
for i = 1:4
	subplot(2,2,i);
	imagesc(reshape(X(i,1,:),nRows,nCols));
	colormap gray
end
suptitle('Examples of Noisy Xs');

%% Make edgeStruct

adj = latticeAdjMatrix(nRows,nCols);
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
nEdges = edgeStruct.nEdges;

%% Make Xnode, Xedge, infoStruct, initialize weights

% Add bias and Standardize Columns
tied = 1;
Xnode = [ones(nInstances,1,nNodes) UGM_standardizeCols(X,tied)];
nNodeFeatures = size(Xnode,2);

% Make Xedge
sharedFeatures = [1 0];
Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,sharedFeatures);

% Make nodeMap, edgeMap, initial parameter vector
tied = 1;
ising = 1;
[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied);
nParams = length(w);

%% Evaluate with random parameters

figure;
w = randn(nParams,1);
for i = 1:4
	subplot(2,2,i);
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
	nodeBel = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
	imagesc(reshape(nodeBel(:,2),nRows,nCols));
	colormap gray
end
suptitle('Loopy BP node marginals with random parameters');
fprintf('(paused)\n');
pause

%% Train with Loopy Belief Propagation for 3 iterations

maxIter = 3; % Number of passes through the data set

w = zeros(nParams,1);
options.maxFunEvals = maxIter;
if 1
	funObj = @(w)UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP); % Loopy belief propagation training
else
	funObj = @(w)UGM_CRF_PseudoNLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct); % Pseudo-likelihood training
end
w = minFunc(funObj,w,options);

figure;
for i = 1:4
	subplot(2,2,i);
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
	nodeBel = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
	imagesc(reshape(nodeBel(:,2),nRows,nCols));
	colormap gray
end
suptitle('Loopy BP node marginals with truncated minFunc parameters');
fprintf('(paused)\n');
pause

%% Train with Stochastic gradient descent for the same amount of time
stepSize = 1e-4;
w = zeros(nParams,1);
fAvg = 0;
for iter = 1:maxIter*nInstances
	% Compute NLL and Gradient for random training example
	i = ceil(rand*nInstances);
	[f,g] = UGM_CRF_NLL(w,Xnode(i,:,:),Xedge(i,:,:),y(i,:),nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
	
	% Update estimate of function value and parameters
	fAvg = (1/iter)*f + ((iter-1)/iter)*fAvg;
	w = w - stepSize*g;
	
	fprintf('Iter = %d of %d (fAvg = %f)\n',iter,maxIter*nInstances,fAvg);
end

figure;
for i = 1:4
	subplot(2,2,i);
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
	nodeBel = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
	imagesc(reshape(nodeBel(:,2),nRows,nCols));
	colormap gray
end
suptitle('Loopy BP node marginals with truncated SGD parameters');
