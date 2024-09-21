clear all
close all

%% Make Adjacency Matrix and EdgeStruct
nStates = [2 2 2 2 2 3 3 3 3]; % Number of states that each node can take
maxState = max(nStates); % Maximum number of states that any node can take
nNodes = length(nStates); % Total number of nodes

adj = zeros(nNodes); % Symmetric {0,1} matrix containing edges
adj(1,2) = 1;
adj(2,3) = 1;
adj(3,4) = 1;
adj(3,5) = 1;
adj(5,6) = 1;
adj(5,7) = 1;
adj(6,8) = 1;
adj = adj+adj';

% Make structure that tracks edge information
edgeStruct = UGM_makeEdgeStruct(adj,nStates); 

%% Demo of how to use the edgeStruct

fprintf('Here was the original adjacency matrix:\n');
adj
fprintf('********** (paused) **********\n\n');pause

fprintf('Here is the list of "edgeEnds":\n(each row gives the two nodes associated with an edge)\n');
edgeStruct.edgeEnds
fprintf('********** (paused) **********\n\n');pause

fprintf('Here is the number of edges:\n');
nEdges = edgeStruct.nEdges
fprintf('********** (paused) **********\n\n');pause

edge = 4;
fprintf('Here are the nodes associated with edge %d\n',edge);
nodes = edgeStruct.edgeEnds(edge,:)
fprintf('********** (paused) **********\n\n');pause

node = 3;
fprintf('Here are the edge numbers associated with node %d\n',node);
edges = UGM_getEdges(node,edgeStruct)
fprintf('********** (paused) **********\n\n');pause

fprintf('These are the edgeEnds associated with these edges\n');
edgeEnds = edgeStruct.edgeEnds(edges,:)
fprintf('********** (paused) **********\n\n');pause

fprintf('Here are the neighbors of node %d\n',node);
neighbors = edgeEnds(edgeEnds ~= node)
fprintf('********** (paused) **********\n\n');pause

%% Make the non-negative node and edge potentials (in this case generated uniformly on [0,1])

fprintf('Here is the number of states that each node can take:\n');
edgeStruct.nStates
fprintf('********** (paused) **********\n\n');pause

% Make (non-negative) potential of each node taking each state
nodePot = zeros(nNodes,maxState); 
for n = 1:nNodes
	for s = 1:nStates(n)
		nodePot(n,s) = rand;
	end
end

% Make (non-negative) potential of each edge taking each state combination
edgePot = zeros(maxState,maxState,edgeStruct.nEdges);
for e = 1:nEdges
	n1 = edgeStruct.edgeEnds(e,1);
	n2 = edgeStruct.edgeEnds(e,2);
	for s1 = 1:nStates(n1)
		for s2 = 1:nStates(n2)
			edgePot(s1,s2,e) = rand;
		end
	end
end

fprintf('Here are the random (non-negative) node potentials we generated...\n');
nodePot
fprintf('********** (paused) **********\n\n');pause

fprintf('Here are the random (non-negative) edge potentials we generated...\n');
edgePot
fprintf('********** (paused) **********\n\n');pause

%% Do decoding, inference, and sampling

fprintf('Decoding: Finding configuration of variables with highest potentials...\n');
MAP = UGM_Decode_Tree(nodePot,edgePot,edgeStruct)
fprintf('********** (paused) **********\n\n');pause

fprintf('Inference: Finding unary and pairwise probabilities, and normalizing constant...\n');
[nodeProbs,edgeProbs,logZ] = UGM_Infer_Tree(nodePot,edgePot,edgeStruct);
nodeProbs
Z = exp(logZ)
fprintf('********** (paused) **********\n\n');pause

fprintf('Sampling: Generating samples according to distribution...\n');
samples = UGM_Sample_Tree(nodePot,edgePot,edgeStruct);
fprintf('Plotting...\n');
figure;
imagesc(samples);
fprintf('********** (paused) **********\n\n');pause

%% Use the samples to train an MRF with untied parameters

Y = samples;
nSamples = size(Y,1);

% Make the nodeMap, edgeMap, and initial parameter vector
ising = 0; % Use full potentials
tied = 0; % Each node/edge has its own parameters
[nodeMap,edgeMap,w] = UGM_makeMRFmaps(edgeStruct,ising,tied);

fprintf('The total number of parameters is:\n');
nParams = length(w)
fprintf('********** (paused) **********\n\n');pause

fprintf('Here are the parameter numbers that will be used to make the node potentials:\n(0 means no parameter)\n');
nodeMap
fprintf('********** (paused) **********\n\n');pause

fprintf('Here are the parameter numbers that will be used to make the edge potentials:\n(0 means no parameter)\n');
edgeMap
fprintf('********** (paused) **********\n\n');pause

fprintf('Here are the sufficient statistics for each parameter:\n');
suffStat = UGM_MRF_computeSuffStat(Y,nodeMap,edgeMap,edgeStruct)
fprintf('********** (paused) **********\n\n');pause

fprintf('Estimation: Training untied MRF with full potentials...\n');
options = [];
inferFunc = @UGM_Infer_Tree; % Function used to perform inference in the model
w = minFunc(@UGM_MRF_NLL,w,options,nSamples,suffStat,nodeMap,edgeMap,edgeStruct,inferFunc);
fprintf('********** (paused) **********\n\n');pause

fprintf('Here are the estimated parameters:\n');
w
fprintf('********** (paused) **********\n\n');pause

%% Form the estimated potentials and do decoding/inference/sampling with them
fprintf('Here are the estimated node potentials:\n');
[np,ep] = UGM_MRF_makePotentials(w,nodeMap,edgeMap,edgeStruct);
np
fprintf('********** (paused) **********\n\n');pause

fprintf('Decoding: Finding configuration of variables with highest estimated potentials...\n');
UGM_Decode_Tree(np,ep,edgeStruct)
fprintf('********** (paused) **********\n\n');pause

fprintf('Inference: Finding unary and pairwise estimated probabilities, and normalizing constant...\n');
[nps,eps,logZ] = UGM_Infer_Tree(np,ep,edgeStruct);
nps
Z = exp(logZ)
fprintf('********** (paused) **********\n\n');pause

fprintf('Sampling: Generating samples according to estimated distribution...\n');
samples = UGM_Sample_Tree(np,ep,edgeStruct);
fprintf('Plotting...\n');
figure;
imagesc(samples);
fprintf('********** (paused) **********\n\n');pause

%% Train a tied binary CRF with Ising-like edge potentials

Y(Y==3) = 2; % We'll just do a binary example here
Y = int32(Y); % Mex files need to have integer data type
edgeStruct.nStates(:) = 2;

% Make node features (we will take a bias, plus 5 random ones)
nNodeFeatures = 5;
Xnode = [ones(nSamples,1,nNodes) randn(nSamples,nNodeFeatures,nNodes)];

% Make edge features (we will just use the node features for both nodes)
sharedFeatures = [1 zeros(1,nNodeFeatures)]; % Only the bias is shared
Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,sharedFeatures);

% Make the nodeMap and edgeMap
ising = 1; % Use Ising-like potentials
tied = 1; % Each node/edge shares parameters
[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied);

% Train
fprintf('Estimation: Training tied binary CRF with Ising-like potentials...\n');
w = minFunc(@UGM_CRF_NLL,w,[],Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct,inferFunc);
fprintf('********** (paused) **********\n\n');pause

%% Form the estimated potentials conditioned on the features for an individual training example,
% and do decoding/inference/sampling with them
i = 10;
fprintf('Here are the estimated node potentials for training example %d:\n',i);
[np,ep] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
np
fprintf('********** (paused) **********\n\n');pause

fprintf('Here are the true labels for this training example %d:\n',i);
Y(10,:)'
fprintf('********** (paused) **********\n\n');pause

fprintf('Decoding: Finding configuration of variables with highest conditional potentials...\n');
UGM_Decode_Tree(np,ep,edgeStruct)
fprintf('********** (paused) **********\n\n');pause

fprintf('Inference: Finding unary and pairwise conditional probabilities, and normalizing constant...\n');
[nps,eps,logZ] = UGM_Infer_Tree(np,ep,edgeStruct);
nps
Z = exp(logZ)
fprintf('********** (paused) **********\n\n');pause

fprintf('Sampling: Generating samples according to estimated conditional distribution...\n');
samples = UGM_Sample_Tree(np,ep,edgeStruct);
fprintf('Plotting...\n');
figure;
imagesc(samples);
fprintf('********** (paused) **********\n\n');pause
