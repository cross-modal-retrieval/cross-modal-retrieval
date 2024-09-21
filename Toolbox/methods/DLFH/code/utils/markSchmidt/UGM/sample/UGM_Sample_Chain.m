function  [samples] = UGM_Sample_Chain(nodePot, edgePot, edgeStruct)

assert(edgeStruct.nEdges==size(nodePot,1)-1,'Running chain sampling method on non-chain data');

[nNodes,maxState] = size(nodePot);
nEdges = size(edgePot,3);
edgeEnds = edgeStruct.edgeEnds;
nStates = edgeStruct.nStates;
nSamples = edgeStruct.maxIter;
maximize = 0;

% Forward Pass
alpha = UGM_ChainFwd(nodePot,edgePot,nStates,maximize);

samples = zeros(nSamples,nNodes);
y = zeros(1,nNodes);

for s = 1:nSamples
    % Backward Pass
    y(nNodes) = sampleDiscrete(alpha(nNodes,:));
    for n = nNodes-1:-1:1
        pot_ij = alpha(n,1:nStates(n))'.*edgePot(1:nStates(n),y(n+1),n);
        y(n) = sampleDiscrete(pot_ij./sum(pot_ij));
    end
    samples(s,:) = y;
end
