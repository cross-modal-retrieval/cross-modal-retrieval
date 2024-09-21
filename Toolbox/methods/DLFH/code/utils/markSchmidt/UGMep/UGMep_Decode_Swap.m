function  [y] = UGMep_Decode_Swap(nodeEnergy, edgeEnergy, edgeWeights, edgeStruct, y)
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

% Do Alpha-Beta swaps until convergence
while 1
    y_old = y;
    
    for s1 = 1:maxState
        for s2 = s1+1:maxState
            swapPositions = find(y==s1 | y==s2);
            if ~isempty(swapPositions)
                % Find optimal re-arrangement of nodes assigned to s1 or s2
                fprintf('Swapping %d and %d\n',s1,s2);
                clamped = y;
                clamped(swapPositions) = 0;
                [clampedNE,clampedEW,clampedES] = UGMep_makeClampedEnergy(nodeEnergy,edgeEnergy,edgeWeights,edgeStruct,clamped);
                
                % Remove all other labels
                clampedNE = clampedNE(:,[s1 s2]);
                clampedEE = edgeEnergy([s1 s2],[s1 s2]);
                clampedES.nStates = 2*ones(size(nStates));
                
                ytmp = UGMep_Decode_GraphCut(clampedNE,clampedEE,clampedEW,clampedES);
                
                clampedY = zeros(size(ytmp));
                clampedY(ytmp==1) = s1;
                clampedY(ytmp==2) = s2;
                
                y2 = y;
                y(swapPositions) = clampedY;
            end
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
