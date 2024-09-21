
fprintf('\n\nTraining %s...\n',type);

%% Make EdgeStruct
edgeStruct = UGM_makeEdgeStruct(adjInit,nStates,useMex);
nEdges = size(edgeStruct.edgeEnds,1);
Xedge = UGM_makeEdgeFeatures(X,edgeStruct.edgeEnds);

% Add Node Bias
Xnode = [ones(nInstances,1,nNodes) X];
Xedge = [ones(nInstances,1,nEdges) Xedge];

%% Make nodeMap and edgeMap
[nodeMap,edgeMap] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied,1);
nNodeParams = max(nodeMap(:));
if isempty(edgeMap)
	nEdgeParams = 0;
else
	nEdgeParams = max(edgeMap(:))-nNodeParams;
end
nVars = nNodeParams+nEdgeParams;
weights = zeros(nVars,1);

%% Set up Objective
if strcmp(trainType,'pseudo')
	funObj_sub = @(weights)UGM_CRF_PseudoNLL(weights,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct);
else
	switch trainType
		case 'loopy'
			inferFunc = @UGM_Infer_LBP;
		case 'exact'
			inferFunc = @UGM_Infer_Exact;
		case 'mean'
			inferFunc = @UGM_Infer_MeanField;
		otherwise
			error('Unrecognized trainType: %s\n',trainType);
	end
		funObj_sub = @(weights)UGM_CRF_NLL(weights,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,inferFunc);
end

%% Set up Regularizer and Train
nodePenalty = lambdaNode*ones(nNodeParams,1);
biasParams = nodeMap(:,:,1);
nodePenalty(biasParams(:)) = 0; % Don't penalize node bias
edgePenalty = lambdaEdge*ones(nEdgeParams,1);
if strcmp(edgePenaltyType,'L2')
    % Train with L2-regularization on node and edge parameters
    funObj = @(weights)penalizedL2(weights,funObj_sub,[nodePenalty(:);edgePenalty(:)]);
    weights = minFunc(funObj,zeros(nVars,1));
elseif strcmp(edgePenaltyType,'L1')
    % Train with L2-regularization on node parameters and
    % L1-regularization on edge parameters
    funObjL2 = @(weights)penalizedL2(weights,funObj_sub,[nodePenalty(:);zeros(nEdgeParams,1)]); % L2 on Node Parameters
    funObj = @(weights)nonNegGrad(weights,[zeros(nNodeParams,1);edgePenalty(:)],funObjL2);
    weights = minConf_TMP(funObj,zeros(2*nVars,1),zeros(2*nVars,1),inf(2*nVars,1));
    weights = weights(1:nVars)-weights(nVars+1:end);
else
    % Train with L2-regularization on node parameters and
    % group L1-regularization on edge parameters
    groups = zeros(nVars,1);
    for e = 1:nEdges
		edgeParams = edgeMap(:,:,e,:);
		groups(edgeParams(:)) = e;
    end
    nGroups = length(unique(groups(groups>0)));
    
    funObjL2 = @(weights)penalizedL2(weights,funObj_sub,[nodePenalty(:);zeros(nEdgeParams,1)]); % L2 on Node Parameters
    
    funObj = @(weights)auxGroupLoss(weights,groups,lambdaEdge,funObjL2);
    [groupStart,groupPtr] = groupl1_makeGroupPointers(groups);
    if strcmp(edgePenaltyType,'L1-L2')
        funProj = @(w)auxGroupL2Project(w,nVars,groupStart,groupPtr);
    elseif strcmp(edgePenaltyType,'L1-Linf')
        funProj = @(w)auxGroupLinfProject(w,nVars,groupStart,groupPtr);
    else
        fprintf('Unrecognized edgePenaltyType\n');
        pause;
    end
    
    weights = minConf_SPG(funObj,[zeros(nVars,1);zeros(nGroups,1)],funProj);
    weights = weights(1:nVars);
end

%% Compute Test Error

% Compute Node/Edge Potentials
if nFeatures == 0
	[nodePot,edgePot] = UGM_MRF_makePotentials(weights,nodeMap,edgeMap,edgeStruct);
end

% Compute Error on Test Data
err = 0;
for i = testNdx
	if nFeatures > 0
		[nodePot,edgePot] = UGM_CRF_makePotentials(weights,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
	end
	
    if strcmp(testType,'exact')
        [nodeBel,edge,logZ] = UGM_Infer_Exact(nodePot,edgePot,edgeStruct);
    elseif strcmp(testType,'loopy')
        [nodeBel,edge,logZ] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
    else
        fprintf('Unrecognized testType: %s\n',testType);
        pause;
    end
    [margConf yMaxMarg] = max(nodeBel,[],2);
    err = err + sum(yMaxMarg' ~= y(i,:));
end
err = err/(length(testNdx)*nNodes);


fprintf('Error Rate (%s): %.3f\n',type,err);

% Find active edges
adjFinal = zeros(nNodes);
for e = 1:nEdges
	edgeParams = edgeMap(:,:,e,:);
	params = edgeParams(edgeParams(:)~=0);
    if any(abs(weights(params)) > 1e-4)
        n1 = edgeStruct.edgeEnds(e,1);
        n2 = edgeStruct.edgeEnds(e,2);
        adjFinal(n1,n2) = 1;
        adjFinal(n2,n1) = 1;
	end
end
if subDisplay
    figure;hold on;
    drawGraph(adjFinal);
    title(sprintf('%s (err = %.3f)',type,err));
    pause;
end