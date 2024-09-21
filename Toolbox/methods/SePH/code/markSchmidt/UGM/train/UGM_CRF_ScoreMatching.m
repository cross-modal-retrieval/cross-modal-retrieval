function [obj,g] = UGM_CRF_ScoreMatching(w,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct)

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
    
    for n = 1:nNodes
        % Find Neighbors
        edges = UGM_getEdges(n,edgeStruct);
        
        % Compute Probability of Each State with Neighbors Fixed
        pot = nodePot(n,1:nStates(n));
        for e = edges
            n1 = edgeEnds(e,1);
            n2 = edgeEnds(e,2);
            
            if n == edgeEnds(e,1)
                ep = edgePot(1:nStates(n),Y(i,n2),e).';
            else
                ep = edgePot(Y(i,n1),1:nStates(n),e);
            end
            pot = pot .* ep;
        end
        Z = sum(pot);
        nodeBel = pot/Z;
        
        % Update objective
        for s = 1:nStates(n)
            if s == Y(i,n)
                obs = 1;
            else
                obs = 0;
            end
            obj = obj + (nodeBel(s)-obs)^2;
        end
        
        %% Update Gradient
        if nargout > 1
            
            % Update Gradient of Node Weights
            for s = 1:nStates(n)
                for f = 1:nNodeFeatures
                    if nodeMap(n,s,f) > 0
                        if s == Y(i,n)
                            obs = 1;
                        else
                            obs = 0;
                        end
                                                
                        for s2 = 1:nStates(s)
                            if s2 == Y(i,n)
                                obs2 = 1;
                            else
                                obs2 = 0;
                            end
                            
                            if s2 == s
                                g(nodeMap(n,s,f)) = g(nodeMap(n,s,f)) + 2*(nodeBel(s)-obs)*Xnode(i,f,n)*(pot(s)/Z - pot(s)^2/Z^2);
                            else
                                g(nodeMap(n,s,f)) = g(nodeMap(n,s,f)) - 2*(nodeBel(s2)-obs2)*Xnode(i,f,n)*pot(s)*pot(s2)/Z^2;
                            end
                        end
                        
                    end
                end
            end
            
            % Update Gradient of Edge Weights
            for e = edges
                
                n1 = edgeEnds(e,1);
                n2 = edgeEnds(e,2);
                
                for s = 1:nStates(n)
                    if n == n1
                        s1 = s;
                        neigh = n2;
                        s2 = Y(i,neigh);
                    else
                        s2 = s;
                        neigh = n1;
                        s1 = Y(i,neigh);
                    end
                    for f = 1:nEdgeFeatures
                        if edgeMap(s1,s2,e,f) > 0
                            if s == Y(i,n)
                                obs = 1;
                            else
                                obs = 0;
                            end
                            
                            for sAlt = 1:nStates(n)
                                if sAlt == Y(i,n)
                                    obs2 = 1;
                                else
                                    obs2 = 0;
                                end
                                
                                if sAlt == s
                                    g(edgeMap(s1,s2,e,f)) = g(edgeMap(s1,s2,e,f)) + 2*(nodeBel(s)-obs)*Xedge(i,f,e)*(pot(s)/Z - pot(s)^2/Z^2);
                                else
                                    g(edgeMap(s1,s2,e,f)) = g(edgeMap(s1,s2,e,f)) - 2*(nodeBel(sAlt)-obs2)*Xedge(i,f,e)*pot(s)*pot(sAlt)/Z^2;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
