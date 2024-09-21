function  [y] = UGM_Decode_GraphCut(nodeEnergy, edgeEnergy, edgeWeights,edgeStruct)
% INPUT
% nodePot(node,class)
% edgePot(class,class,edge) where e is referenced by V,E (must be the same
% between feature engine and inference engine)
%
% OUTPUT
% nodeLabel(node)

nNodes = size(nodeEnergy,1);
edgeEnds = edgeStruct.edgeEnds;
nEdges = size(edgeEnds,1);
V = edgeStruct.V;
E = edgeStruct.E;
nStates = edgeStruct.nStates;

assert(all(nStates == 2),'Graph Cuts only implemented for binary graphs');

% Check Sub-Modularity Condition
assert(edgeEnergy(1,1)+edgeEnergy(2,2)<=edgeEnergy(1,2)+edgeEnergy(2,1)+1e-15,...
	'Graph Cuts only implemented for sub-modular potentials\n');

% Move energy from edges to nodes
if edgeStruct.useMex
	UGMep_Decode_GraphCutC(nodeEnergy,edgeEnergy,edgeWeights,int32(edgeEnds-1));
else
	for e = 1:nEdges
		n1 = edgeEnds(e,1);
		n2 = edgeEnds(e,2);
		nodeEnergy(n1,2) = nodeEnergy(n1,2) + edgeWeights(e)*(edgeEnergy(2,1) - edgeEnergy(1,1));
		nodeEnergy(n2,2) = nodeEnergy(n2,2) + edgeWeights(e)*(edgeEnergy(2,2) - edgeEnergy(2,1));
	end
end

% Make Graph
sCapacities = zeros(nNodes,1);
tCapacities = zeros(nNodes,1);
ndx = nodeEnergy(:,1) < nodeEnergy(:,2);
sCapacities(ndx) = nodeEnergy(ndx,2) - nodeEnergy(ndx,1);
tCapacities(~ndx) = nodeEnergy(~ndx,1) - nodeEnergy(~ndx,2);
eCapacities = edgeWeights*(edgeEnergy(1,2)+edgeEnergy(2,1)-edgeEnergy(1,1)-edgeEnergy(2,2));
eCapacities = max(0,eCapacities(:));

%% Solve Max-Flow Problem

T = sparse([sCapacities tCapacities]);
A = sparse(edgeEnds(:,1),edgeEnds(:,2),eCapacities,nNodes,nNodes,nEdges);
[flow,y] = maxflow(A,T);
y = y+1;
end

function assert(pred, str)
% ASSERT Raise an error if the predicate is not true.
% assert(pred, string)

if nargin<2, str = ''; end

if ~pred
	s = sprintf('assertion violated: %s', str);
	error(s);
end
end
