function [edgeStruct] = UGM_makeEdgeStruct(adj,nStates,useMex,maxIter)
% [edgeStruct] = UGM_makeEdgeStruct(adj,nStates,useMex,maxIter)
%
% adj - nNodes by nNodes adjacency matrix (assumed symmetric)
%

if nargin < 3
    useMex = 1;
end
if nargin < 4
    maxIter = 100;
end

[nNodes,nNodes2] = size(adj);
UGM_assert(nNodes==nNodes2,'Adjacency matrix must be square');
nNodes = int32(nNodes);

[i j] = ind2sub([nNodes nNodes],find(adj));
nEdges = length(i)/2;
edgeEnds = zeros(nEdges,2,'int32');
eNum = 0;
for e = 1:length(i)
   if j(e) < i(e)
       edgeEnds(eNum+1,:) = [j(e) i(e)];
       eNum = eNum+1;
   end
end
assert(eNum==nEdges,'Something is wrong with the adjacency matrix (possibly non-symmetric, or non-zero on diagonals)');

[V,E] = UGM_makeEdgeVE(edgeEnds,nNodes,useMex);


edgeStruct.edgeEnds = edgeEnds;
edgeStruct.V = V;
edgeStruct.E = E;
edgeStruct.nNodes = nNodes;
edgeStruct.nEdges = size(edgeEnds,1);

% Handle other arguments
assert(~any(nStates<2),'Each node must have at least 2 states');
if isscalar(nStates)
   nStates = repmat(nStates,[double(nNodes) 1]);
end
UGM_assert(length(nStates)==nNodes,'nStates vector must have nNodes elements');
edgeStruct.nStates = int32(nStates(:));
edgeStruct.useMex = useMex;
edgeStruct.maxIter = int32(maxIter);


