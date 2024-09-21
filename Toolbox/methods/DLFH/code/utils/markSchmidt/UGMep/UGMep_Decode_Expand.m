function  [y] = UGMep_Decode_Expand(nodeEnergy, edgeEnergy, edgeWeights, edgeStruct, y)
% INPUT
% nodePot(node,class)
% edgePot(class,class,edge) where e is referenced by V,E (must be the same
% between feature engine and inference engine)
%
% OUTPUT
% nodeLabel(node)

[nNodes,maxState] = size(nodeEnergy);
edgeEnds = edgeStruct.edgeEnds;
nEdges = size(edgeEnds,1);
V = edgeStruct.V;
E = edgeStruct.E;
nStates = edgeStruct.nStates;

% Initialize
if nargin < 5
	[junk y] = min(nodeEnergy,[],2);
end
if edgeStruct.useMex
	energy = UGMep_EnergyC(int32(y-1),nodeEnergy,edgeEnergy,edgeWeights,int32(edgeEnds-1));
else
	energy = UGMep_Energy(y,nodeEnergy,edgeEnergy,edgeWeights,edgeEnds);
end
fprintf('energy = %f\n',energy);

% Do Alpha-Expansions until convergence
while 1
	y_old = y;
	
	for s = 1:maxState
		swapPositions = find(y~=s);
		if ~isempty(swapPositions)
			fprintf('Expanding %d\n',s);
			clamped = y;
			clamped(swapPositions) = 0;
			[clampedNE,clampedEW,clampedES] = UGMep_makeClampedEnergy(nodeEnergy,edgeEnergy,edgeWeights,edgeStruct,clamped);
			
			if edgeStruct.useMex
				[modifiedNE,modifiedEE] = UGMep_Decode_AlphaExpansionC(clampedNE,edgeEnergy,clampedEW,int32(clampedES.edgeEnds-1),int32(s-1),int32(y-1),int32(swapPositions-1));
			else
				nClampedNodes = size(clampedNE,1);
				modifiedNE = zeros(nClampedNodes,2);
				for n = 1:nClampedNodes
					modifiedNE(n,:) = [clampedNE(n,s) clampedNE(n,y(swapPositions(n)))];
				end
				nClampedEdges = size(clampedES.edgeEnds,1);
				modifiedEE = zeros(2,2,nClampedEdges);
				for e = 1:nClampedEdges
					n1 = clampedES.edgeEnds(e,1);
					n2 = clampedES.edgeEnds(e,2);
					modifiedEE(:,:,e) = clampedEW(e)*[edgeEnergy(s,s) edgeEnergy(s,y(swapPositions(n2)))
						edgeEnergy(y(swapPositions(n1)),s) edgeEnergy(y(swapPositions(n1)),y(swapPositions(n2)))];
				end
			end
			clampedES.nStates(:) = 2;
			
			% Decreasing energy of state (2,2) so that edges are
			% sub-modular (does nothing if edges are metric)
			modifiedEE(2,2,:) = min(modifiedEE(2,2,:),modifiedEE(1,2,:)+modifiedEE(2,1,:)-modifiedEE(1,1,:));
			
			ytmp = UGMep_Decode_GraphCutFull(modifiedNE,modifiedEE,clampedES);
			swapInd = find(ytmp == 1);
			
			y(swapPositions(swapInd)) = s;
			
		end
	end
	
	energy_old = energy;
	if edgeStruct.useMex
		energy = UGMep_EnergyC(int32(y-1),nodeEnergy,edgeEnergy,edgeWeights,int32(edgeEnds-1));
	else
		energy = UGMep_Energy(y,nodeEnergy,edgeEnergy,edgeWeights,edgeEnds);
	end
	fprintf('energy = %f, changes = %d\n',energy,sum(y_old~=y));
	if all(y==y_old)
		break;
	end
	if energy-energy_old == 0
		break;
	end
end
