function [obj,g] = UGM_CRF_Piecewise(w,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct)

[nNodes,maxState] = size(nodeMap);
nNodeFeatures = size(Xnode,2);
nEdgeFeatures = size(Xedge,2);
nEdges = edgeStruct.nEdges;
edgeEnds = edgeStruct.edgeEnds;
V = edgeStruct.V;
E = edgeStruct.E;
nStates = edgeStruct.nStates;

nInstances = size(Y,1);
obj = 0;
g = zeros(size(w));

for i = 1:nInstances
    % Make potentials
    if edgeStruct.useMex
        [nodePot,edgePot] = UGM_CRF_makePotentialsC(w,Xnode,Xedge,nodeMap,edgeMap,nStates,edgeEnds,int32(i));
    else
        [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
    end
    
    for e = 1:nEdges
        n1 = edgeEnds(e,1);
        n2 = edgeEnds(e,2);
        
        for s1 = 1:nStates(n1)
            for s2 = 1:nStates(n2)
                pot(s1,s2) = nodePot(n1,s1)*nodePot(n2,s2)*edgePot(s1,s2,e);
            end
        end
        Z = sum(pot(:));
        
        % Update Objective
        obj = obj - log(pot(Y(i,n1),Y(i,n2))) + log(Z);
        
        if nargout > 1
            nodeBel = sum(pot,2)/Z;
            for s = 1:nStates(n1)
                for f = 1:nNodeFeatures
                    if nodeMap(n1,s,f) > 0
                        if s == Y(i,n1)
                            obs = 1;
                        else
                            obs = 0;
                        end
                        g(nodeMap(n1,s,f)) = g(nodeMap(n1,s,f)) + Xnode(i,f,n1)*(nodeBel(s) - obs);
                    end
                end
            end
            nodeBel = sum(pot)/Z;
            for s = 1:nStates(n2)
                for f = 1:nNodeFeatures
                    if nodeMap(n2,s,f) > 0
                        if s == Y(i,n2)
                            obs = 1;
                        else
                            obs = 0;
                        end
                        g(nodeMap(n2,s,f)) = g(nodeMap(n2,s,f)) + Xnode(i,f,n2)*(nodeBel(s) - obs);
                    end
                end
            end
            edgeBel = pot/Z;
            for s1 = 1:nStates(n1)
                for s2 = 1:nStates(n2)
                    for f = 1:nEdgeFeatures
                        if edgeMap(s1,s2,e,f) > 0
                            if s1 == Y(i,n1) && s2 == Y(i,n2)
                                obs = 1;
                            else
                                obs = 0;
                            end
                            g(edgeMap(s1,s2,e,f)) = g(edgeMap(s1,s2,e,f)) + Xedge(i,f,e)*(edgeBel(s1,s2) - obs);
                        end
                    end
                end
            end
        end
    end
end
