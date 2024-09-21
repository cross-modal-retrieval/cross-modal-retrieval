function  [nodeLabels] = UGM_Decode_Chain(nodePot, edgePot, edgeStruct)
% optimal decoding of chain-structured graphical model

assert(edgeStruct.nEdges==size(nodePot,1)-1,'Running chain decoding method on non-chain data');
if edgeStruct.useMex
	nodeLabels = UGM_Decode_ChainC(nodePot,edgePot,edgeStruct.nStates);
else
	[nNodes,maxState] = size(nodePot);
	nEdges = size(edgePot,3);
	edgeEnds = edgeStruct.edgeEnds;
	nStates = edgeStruct.nStates;
	maximize = 1;
	
	% Forward Pass
	[alpha,kappa,mxState] = UGM_ChainFwd(nodePot,edgePot,nStates,maximize);
	
	% Backward Pass
	nodeLabels = zeros(nNodes,1);
	[mxPot nodeLabels(nNodes)] = max(alpha(nNodes,:));
	for n = nNodes-1:-1:1
		nodeLabels(n) = mxState(n+1,nodeLabels(n+1));
	end
end