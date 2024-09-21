
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

%% Train with Pseudo-likelihood

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

%% Train with Score Matching

w = zeros(nParams,1);
funObj = @(w)UGM_CRF_ScoreMatching(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct);
fprintf('Training with score matching...\n');
w = minFunc(funObj,w);

fprintf('ICM Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('ICM Decoding with score-matching parameters');
fprintf('(paused)\n');
pause

%% Train with Piecewise

w = zeros(nParams,1);
funObj = @(w)UGM_CRF_Piecewise(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct);
fprintf('Training with piecewise likelihood...\n');
w = minFunc(funObj,w);

fprintf('ICM Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('ICM Decoding with piecewise parameters');
fprintf('(paused)\n');
pause

%% Train with Minimum Probability Flow

w = zeros(nParams,1);
funObj = @(w)UGM_CRF_MinFlow(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct);
fprintf('Training with minimum probability flow...\n');
w = minFunc(funObj,w,struct('numDiff',1));

fprintf('ICM Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('ICM Decoding with min probability flow parameters');
fprintf('(paused)\n');
pause

%% Train with Product of Marginals

w = zeros(nParams,1);
funObj = @(w)UGM_CRF_ProductMarginal(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
fprintf('Training with product of marginals...\n');
w = minFunc(funObj,w,struct('numDiff',1));

fprintf('ICM Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('ICM Decoding with product of marginals parameters');
fprintf('(paused)\n');
pause

%% Make Blocks

if 1 % Make 2 blocks, consisting of 2 trees that cover the nodes
    nodeNums = reshape(1:nNodes,nRows,nCols);
    blocks1 = zeros(nNodes/2,1);
    blocks2 = zeros(nNodes/2,1);
    b1Ind = 0;
    b2Ind = 0;
    for j = 1:nCols
        if mod(j,2) == 1
            blocks1(b1Ind+1:b1Ind+nCols-1) = nodeNums(1:nCols-1,j);
            b1Ind = b1Ind+nCols-1;
            
            blocks2(b2Ind+1) = nodeNums(nRows,j);
            b2Ind = b2Ind+1;
        else
            blocks1(b1Ind+1) = nodeNums(1,j);
            b1Ind = b1Ind+1;
            
            blocks2(b2Ind+1:b2Ind+nCols-1) = nodeNums(2:nCols,j);
            b2Ind = b2Ind+nCols-1;
        end
    end
    blocks = {blocks1;blocks2};
else
    for n = 1:nNodes
        blocks{n} = n; % Equivalent to standard pseudo-likelihood
    end
end

%% Train with Block-Pseudolikelihood

w = zeros(nParams,1);
funObj = @(w)UGM_CRF_Block_PseudoNLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,blocks,@UGM_Infer_Tree);
fprintf('Training with block pseudo-likelihood...\n');
w = minFunc(funObj,w);

fprintf('ICM Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('ICM Decoding with block pseudo-likelihood parameters');
fprintf('(paused)\n');
pause

%% Now try with non-negative edge features and sub-modular restriction

sharedFeatures = [1 0];
Xedge = UGM_makeEdgeFeaturesInvAbsDif(Xnode,edgeStruct.edgeEnds,sharedFeatures);
nEdgeFeatures = size(Xedge,2);

[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied);
nParams = length(w);

funObj = @(w)UGM_CRF_PseudoNLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct); % Make objective with new Xedge/edgeMap
UB = [inf;inf;inf;inf]; % No upper bound on parameters
LB = [-inf;-inf;0;0]; % No lower bound on node parameters, edge parameters must be non-negative 
fprintf('Training with pseudo-likelihood and sub-modular constraint...\n');
w = minConf_TMP(funObj,w,LB,UB)

fprintf('Graph Cuts Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_GraphCut(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('GraphCut Decoding with constrained pseudo-likelihood parameters');
fprintf('(paused)\n');
pause

%% Now try with loopy belief propagation for approximate inference

w = zeros(nParams,1);
fprintf('Training with Bethe free energy and sub-modular constraint...\n');
funObj = @(w)UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
w = minConf_TMP(funObj,w,LB,UB);

fprintf('Graph Cuts Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_GraphCut(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title('GraphCut Decoding with constrained loopy BP parameters');
fprintf('(paused)\n');
pause
