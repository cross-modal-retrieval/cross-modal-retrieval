function [nodeMap,edgeMap,w] = UGM_makeMRFmaps(edgeStruct,ising,tied,paramLastState)
% [nodeMap,edgeMap,w] = UGM_makeMRFmaps(edgeStruct,ising,tied,paramLastState)
%
% Recent change: no longer requires nNodes as input parameter

if nargin < 4
    paramLastState = 0;
end

nNodes = edgeStruct.nNodes;
nEdges = edgeStruct.nEdges;
nStates = edgeStruct.nStates;
maxState = max(nStates);

UGM_assert(~tied || min(nStates)==maxState,'UGM_makeMRFmaps assumes that all nodes have the same number of states for tied models');

nodeMap = zeros(nNodes,maxState,'int32');
if tied
	nStates = nStates(1);
    if paramLastState
        for s = 1:nStates
            nodeMap(:,s) = s;
        end
        nNodeParams = nStates;
    else
        for s = 1:nStates-1
            nodeMap(:,s) = s;
        end
        nNodeParams = nStates-1;
    end
else
	fNum = 0;
	for n = 1:nNodes
        if paramLastState
            for s = 1:nStates(n)
                fNum = fNum+1;
                nodeMap(n,s) = fNum;
            end
        else
            for s = 1:nStates(n)-1
                fNum = fNum+1;
                nodeMap(n,s) = fNum;
            end
        end
	end
	nNodeParams = fNum;
end

edgeMap = zeros(maxState,maxState,nEdges,'int32');
if tied
	switch ising
		case 1
			for s = 1:nStates
				edgeMap(s,s,:) = nNodeParams+1;
			end
		case 2
			for s = 1:nStates
				edgeMap(s,s,:) = nNodeParams+s;
			end
		case 0
			s = 1;
			for s1 = 1:nStates
				for s2 = 1:nStates
					edgeMap(s1,s2,:) = nNodeParams+s;
					s = s+1;
				end
			end
	end
else
	switch ising
		case 1
			for e = 1:nEdges
				n1 = edgeStruct.edgeEnds(e,1);
				n2 = edgeStruct.edgeEnds(e,2);
				for s = 1:min(nStates(n1),nStates(n2))
					edgeMap(s,s,e) = nNodeParams+e;
				end
			end
		case 2
			se = 1;
			for e = 1:nEdges
				n1 = edgeStruct.edgeEnds(e,1);
				n2 = edgeStruct.edgeEnds(e,2);
				for s = 1:min(nStates(n1),nStates(n2))
					edgeMap(s,s,e) = nNodeParams+se;
					se = se+1;
				end
			end
		case 0
			sse = 1;
			for e = 1:nEdges
				n1 = edgeStruct.edgeEnds(e,1);
				n2 = edgeStruct.edgeEnds(e,2);
				for s1 = 1:nStates(n1)
					for s2 = 1:nStates(n2)
						edgeMap(s1,s2,e) = nNodeParams+sse;
						sse = sse+1;
					end
				end
			end
	end
end
if ~isempty(edgeMap)
    w = zeros(max(edgeMap(:)),1);
else
    w = zeros(max(nodeMap(:)),1);
end