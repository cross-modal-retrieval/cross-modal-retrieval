function  [y] = UGMep_Decode_ExapndShrink(nodeEnergy, edgeEnergy, edgeWeights, edgeStruct, betaSelect, y)
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
	betaSelect = 0;
end
if nargin < 6
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
	
	for s1 = 1:maxState
		switch betaSelect
			case 0 % Basic alpha-expansion
				y = applyMove(y,s1,s1,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct);
			case 1 % Randomized selection of beta
				ind = [1:s1-1 s1+1:maxState];
				y = applyMove(y,s1,ind(ceil(rand*(maxState-1))),nodeEnergy,edgeEnergy,edgeWeights,edgeStruct);
			case 2 % Iterate over choices of beta
				for s2 = 1:maxState
					y = applyMove(y,s1,s2,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct);
				end
			case 3 % Iterate over choices of alpha
				for s2 = 1:maxState
					y = applyMove(y,s2,s1,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct);
				end
			case 4 % Set beta = alpha-1
				y = applyMove(y,s1,max(s1-1,1),nodeEnergy,edgeEnergy,edgeWeights,edgeStruct);
			case 5 % Set beta = alpha+1
				y = applyMove(y,s1,min(s1+1,maxState),nodeEnergy,edgeEnergy,edgeWeights,edgeStruct);
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

end

function y = applyMove(y,s1,s2,nodeEnergy,edgeEnergy,edgeWeights,edgeStruct)
[nNodes,maxState] = size(nodeEnergy);
edgeEnds = edgeStruct.edgeEnds;
nEdges = size(edgeEnds,1);

fprintf('Expanding %d, shrinking with %d\n',s1,s2);
modifiedNE = zeros(nNodes,2);
modifiedEE = zeros(2,2,nEdges);
if edgeStruct.useMex
	UGMep_Decode_ExpandShrinkC(int32(y-1),int32(s1-1),int32(s2-1),nodeEnergy,edgeEnergy,edgeWeights,int32(edgeEnds-1),modifiedNE,modifiedEE);
	
	% Make Graph
	sCapacities = zeros(nNodes,1);
	tCapacities = zeros(nNodes,1);
	ndx = modifiedNE(:,1) < modifiedNE(:,2);
	sCapacities(ndx) = modifiedNE(ndx,2) - modifiedNE(ndx,1);
	tCapacities(~ndx) = modifiedNE(~ndx,1) - modifiedNE(~ndx,2);
	eCapacities = modifiedEE(1,2,:)+modifiedEE(2,1,:)-modifiedEE(1,1,:)-modifiedEE(2,2,:);
	eCapacities = max(0,eCapacities(:));
	
	%% Solve Max-Flow Problem
	T = sparse([sCapacities tCapacities]);
	A = sparse(double(edgeEnds(:,1)),double(edgeEnds(:,2)),eCapacities,nNodes,nNodes,nEdges);
	[flow,ytmp] = maxflow(A,T);
	ytmp = ytmp+1;
else
	
	for n = 1:nNodes
		if y(n) == s1
			modifiedNE(n,:) = [nodeEnergy(n,s1) nodeEnergy(n,s2)]; % Keep alpha or shrink with beta
		else
			modifiedNE(n,:) = [nodeEnergy(n,s1) nodeEnergy(n,y(n))]; % Keep current state or switch to alpha
		end
	end
	for e = 1:nEdges
		n1 = edgeEnds(e,1);
		n2 = edgeEnds(e,2);
		y1 = y(n1);
		y2 = y(n2);
		if y1 == s1
			y1 = s2;
		end
		if y2 == s1
			y2 = s2;
		end
		modifiedEE(:,:,e) = edgeWeights(e)*[edgeEnergy(s1,s1) edgeEnergy(s1,y2)
			edgeEnergy(y1,s1) edgeEnergy(y1,y2)];
		
		% Change energy to make sub-modular (does nothing if edges are metric)
		if y(n1) == s1 && y(n2) == s1
			% (alpha,alpha) edge, decrease energy of staying at (alpha,alpha)
			modifiedEE(1,1,e) = min(modifiedEE(1,1,e),modifiedEE(1,2,e)+modifiedEE(2,1,e)-modifiedEE(2,2,e));
		elseif y(n1) == s1
			% (alpha,~alpha) edge, increase energy of changing to (~alpha,alpha)
			modifiedEE(2,1,e) = max(modifiedEE(2,1,e),modifiedEE(1,1,e)+modifiedEE(2,2,e)-modifiedEE(1,2,e));
		elseif y(n2) == s1
			% (~alpha,alpha) edge, increase energy of changing to (alpha,~alpha)
			modifiedEE(1,2,e) = max(modifiedEE(1,2,e),modifiedEE(1,1,e)+modifiedEE(2,2,e)-modifiedEE(2,1,e));
		else
			% (~alpha,~alpha) edge, decrease of energy of staying at (~alpha,~alpha)
			modifiedEE(2,2,e) = min(modifiedEE(2,2,e),modifiedEE(1,2,e)+modifiedEE(2,1,e)-modifiedEE(1,1,e));
		end
	end
	modifiedES = edgeStruct;
	modifiedES.nStates(:) = 2;
	ytmp = UGMep_Decode_GraphCutFull(modifiedNE,modifiedEE,modifiedES);
end
y(ytmp == 2 & y == s1) = s2;
y(ytmp == 1) = s1;
end