
%% Load Noisy X
clear all
close all

rand('state',0);
randn('state',0);

load X.mat

y = int32(1+X);

figure;
imagesc(X);
colormap gray
title('Original X');

figure;
X = X + randn(size(X))/2;
imagesc(X);
colormap gray
title('Noisy X');

[nRows,nCols] = size(X);
nNodes = nRows*nCols;
nStates = 2;
y = reshape(y,[1 1 nNodes]);
X = reshape(X,1,1,nNodes);

%% Make edgeStruct

adj = latticeAdjMatrix(nRows,nCols);
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
%edgeStruct.useMex = 0;
nEdges = edgeStruct.nEdges;

%% Make Xnode, Xedge, nodeMap, edgeMap, initialize weights

% Add bias and Standardize Columns
tied = 1;
Xnode = [ones(1,1,nNodes) UGM_standardizeCols(X,tied)];
nNodeFeatures = size(Xnode,2);

% Make Xedge
sharedFeatures = [1 0];
Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,sharedFeatures);
nEdgeFeatures = size(Xedge,2);

% Make nodeMap, edgeMap, initialize weights
ising = 1;
tied = 1;
[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied);
nParams = length(w);

%% Evaluate with random parameters

figure;
for i = 1:4
    fprintf('ICM Decoding with random parameters (%d of 4)...\n',i);
    subplot(2,2,i);
    w = randn(nParams,1);
    [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
    yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
    imagesc(reshape(yDecode,nRows,nCols));
    colormap gray
end
suptitle('ICM Decoding with random parameters');
fprintf('(paused)\n');
pause

%% Train with pseudo-likelihood

w = zeros(nParams,1);
funObj = @(w)UGM_CRF_PseudoNLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct);
fprintf('Training with pseudo-likelihood...\n');
w = minFunc(funObj,w);

% Evaluate with learned parameters

fprintf('ICM Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('ICM Decoding with pseudo-likelihood parameters');
fprintf('(paused)\n');
pause

%% Train with mean field approximation for approximate inference

w = zeros(nParams,1);
fprintf('Training with Gibbs mean-field free energy...\n');
funObj = @(w)UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_MeanField);
w = minFunc(funObj,w);

fprintf('Max of mean-field marginals decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_MeanField);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('Max of mean-field marginals Decoding with mean-field parameters');
fprintf('(paused)\n');
pause

%% Train with loopy belief propagation for approximate inference

w = zeros(nParams,1);
fprintf('Training with Bethe free energy...\n');
funObj = @(w)UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
w = minFunc(funObj,w);

fprintf('LBP Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('LBP Decoding with loopy belief propagation parameters');

