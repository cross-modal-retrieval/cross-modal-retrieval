clear all
close all
load rain.mat

% Make rain labels y, and binary month features X
y = int32(X+1);
[nInstances,nNodes] = size(y);

%% Make edgeStruct
nStates = max(y);
adj = zeros(nNodes);
for i = 1:nNodes-1
    adj(i,i+1) = 1;
end
adj = adj+adj';
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
nEdges = edgeStruct.nEdges;
maxState = max(nStates);

%% Training (no features)

% Make simple bias features
Xnode = ones(nInstances,1,nNodes);
Xedge = ones(nInstances,1,nEdges);

% Make nodeMap, edgeMap, and initialize paramter vector
ising = 0;
tied = 1;
[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied,0);

% Optimize
w = minFunc(@UGM_CRF_NLL,w,[],Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain)

% Example of making potentials for the first training example
instance = 1;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,instance);
nodePot(1,:)
edgePot(:,:,1)
fprintf('(paused)\n');
%pause

% Xnode = Xnode(1:10,:,:);
% Xedge = Xedge(1:10,:,:);
% y = y(1:10,:);
% w = minFunc(@UGM_CRF_ProductMarginal,randn(size(w)),struct('derivativeCheck',1),Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Tree);
% [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,instance);
% nodePot(1,:)
% edgePot(:,:,1)
% fprintf('(paused)\n');
% pause

%% Training (with node features, but no edge features)

% Make simple bias features
nFeatures = 12;
Xnode = zeros(nInstances,nFeatures,nNodes);
for m = 1:nFeatures
    Xnode(months==m,m,:) = 1;
end
Xnode = [ones(nInstances,1,nNodes) Xnode];
nNodeFeatures = size(Xnode,2);

[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied);

% Optimize
w = minFunc(@UGM_CRF_NLL,w,[],Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain)
fprintf('(paused)\n');
pause

%% Training (with edge features)

% Make edge features
sharedFeatures = ones(13,1);
Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,sharedFeatures);
nEdgeFeatures = size(Xedge,2);

[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied);

% Optimize
UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain);
w = minFunc(@UGM_CRF_NLL,w,[],Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain)
fprintf('(paused)\n');
pause

%% Do decoding/infence/sampling in learned model (given features)

% We will look at a case in December
i = 11;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);

decode = UGM_Decode_Chain(nodePot,edgePot,edgeStruct)

[nodeBel,edgeBel,logZ] = UGM_Infer_Chain(nodePot,edgePot,edgeStruct);
nodeBel

samples = UGM_Sample_Chain(nodePot,edgePot,edgeStruct);
figure
imagesc(samples)
title('Samples from CRF model (for December)');
fprintf('(paused)\n');
pause

%% Do conditional decoding/inference/sampling in learned model (given features)

clamped = zeros(nNodes,1);
clamped(1:2) = 2;

condDecode = UGM_Decode_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Decode_Chain)
condNodeBel = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Infer_Chain)
condSamples = UGM_Sample_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Sample_Chain);

figure
imagesc(condSamples)
title('Conditional samples from CRF model (for December)');
fprintf('(paused)\n');
pause

%% Now see what samples in July look like

XtestNode = [1 0 0 0 0 0 0 1 0 0 0 0 0]; % Turn on bias and indicator variable for July
XtestNode = repmat(XtestNode,[1 1 nNodes]);
XtestEdge = UGM_makeEdgeFeatures(XtestNode,edgeStruct.edgeEnds,sharedFeatures);

[nodePot,edgePot] = UGM_CRF_makePotentials(w,XtestNode,XtestEdge,nodeMap,edgeMap,edgeStruct);

samples = UGM_Sample_Chain(nodePot,edgePot,edgeStruct);
figure
imagesc(samples)
title('Samples from CRF model (for July)');
fprintf('(paused)\n');
pause

%% Training with L2-regularization

% Set up regularization parameters
lambda = 10*ones(size(w));
lambda(1) = 0; % Don't penalize node bias variable
lambda(14:17) = 0; % Don't penalize edge bias variable
regFunObj = @(w)penalizedL2(w,@UGM_CRF_NLL,lambda,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain);

% Optimize
nParams = length(w);
w = zeros(nParams,1);
w = minFunc(regFunObj,w);
NLL = UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain)
