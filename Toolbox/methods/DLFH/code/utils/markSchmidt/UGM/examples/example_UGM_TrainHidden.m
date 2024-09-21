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

%% Training with fully observed data

if 0
nFeatures = 12;
Xnode = zeros(nInstances,nFeatures,nNodes);
for m = 1:nFeatures
    Xnode(months==m,m,:) = 1;
end
Xnode = [ones(nInstances,1,nNodes) Xnode];

sharedFeatures = ones(13,1);
Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,sharedFeatures);
nEdgeFeatures = size(Xedge,2);
else
   Xnode = ones(nInstances,1,nNodes);
   Xedge = ones(nInstances,1,nEdges);
end

% Make nodeMap, edgeMap, and initialize paramter vector
ising = 0;
tied = 1;
[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied,0);

% Optimize
w = minFunc(@UGM_CRF_NLL,w,[],Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Chain)

[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,1);
nodeBel = UGM_Infer_Chain(nodePot,edgePot,edgeStruct);

%% Training with missing labels

yMissing = y;
yMissing(rand(numel(y),1) < 0.1) = 0; % Hide the values of a portion of the entries

w = minFunc(@UGM_CRF_NLL_Hidden,zeros(size(w)),[],Xnode,Xedge,yMissing,nodeMap,edgeMap,edgeStruct,@UGM_Infer_Tree);

[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,1);
nodeBel2 = UGM_Infer_Chain(nodePot,edgePot,edgeStruct);
