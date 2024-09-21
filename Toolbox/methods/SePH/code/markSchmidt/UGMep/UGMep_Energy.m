function [energy] = Energy(y,nodeEnergy,edgeEnergy,edgeWeights,edgeEnds)
nNodes = size(nodeEnergy,1);
nEdges = size(edgeEnds,1);
energy = 0;
for n = 1:nNodes
	energy = energy+nodeEnergy(n,y(n));
end
for e = 1:nEdges
	n1 = edgeEnds(e,1);
	n2 = edgeEnds(e,2);
	energy = energy+edgeWeights(e)*edgeEnergy(y(n1),y(n2));
end
end
