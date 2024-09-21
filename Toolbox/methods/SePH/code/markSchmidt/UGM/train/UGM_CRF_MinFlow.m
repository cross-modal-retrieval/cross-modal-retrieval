function [obj,g] = UGM_CRF_MinFlow(w,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct)
% Minimum Probability Flow objective using all neighbors as alternative
% states
% (this function could be made more efficient by using that only a few
% terms change when you change one label)

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
    
    % Compute potential of observed label
    if edgeStruct.useMex
        energyJ = -UGM_LogConfigurationPotentialC(Y(i,:),nodePot,edgePot,edgeStruct.edgeEnds);
    else
        energyJ = -UGM_LogConfigurationPotential(Y(i,:),nodePot,edgePot,edgeStruct.edgeEnds);
    end
    
    y = Y(i,:);
    for n = 1:nNodes
        for s = 1:nStates(n)
            if s ~= Y(i,n)
                y(n) = s;
                
                energy2 = energyJ;
                energy2 = energy2 + log(nodePot(n,Y(i,n)));
                energy2 = energy2 - log(nodePot(n,s));
                for e = UGM_getEdges(n,edgeStruct)
                    n1 = edgeStruct.edgeEnds(e,1);
                    n2 = edgeStruct.edgeEnds(e,2);
                    if n == n1
                        energy2 = energy2 + log(edgePot(Y(i,n),y(n2),e));
                        energy2 = energy2 - log(edgePot(s,y(n2),e));
                    else
                        energy2 = energy2 + log(edgePot(y(n1),Y(i,n),e));
                        energy2 = energy2 - log(edgePot(y(n1),s,e));
                    end
                end
                
                transitionRate = exp((energyJ - energy2)/2);
                obj = obj + transitionRate;
                
                if nargout > 1
                    g = updateGrad(g,transitionRate,i,Xnode,Xedge,Y(i,:),y,nodeMap,edgeMap,edgeStruct);
                end
            end
        end
        y(n) = Y(i,n);
    end
end

end

function g = updateGrad(g,transitionRate,i,Xnode,Xedge,Y,y,nodeMap,edgeMap,edgeStruct)
[nNodes,maxState] = size(nodeMap);
nNodeFeatures = size(Xnode,2);
nEdgeFeatures = size(Xedge,2);
nEdges = edgeStruct.nEdges;
edgeEnds = edgeStruct.edgeEnds;
nStates = edgeStruct.nStates;
for n = 1:nNodes
    for f = 1:nNodeFeatures
        s = Y(i,n);
        if nodeMap(n,s,f) > 0
            g(nodeMap(n,s,f)) = g(nodeMap(n,s,f)) - Xnode(i,f,n)*transitionRate/2;
        end
        s = y(n);
        if nodeMap(n,s,f) > 0
            g(nodeMap(n,s,f)) = g(nodeMap(n,s,f)) + Xnode(i,f,n)*transitionRate/2;
        end
    end
end
for e = 1:nEdges
    n1 = edgeEnds(e,1);
    n2 = edgeEnds(e,2);
   for f = 1:nEdgeFeatures
       s1 = Y(i,n1);
       s2 = Y(i,n2);
       if edgeMap(s1,s2,e,f) > 0
           g(edgeMap(s1,s2,e,f)) = g(edgeMap(s1,s2,e,f)) - Xedge(i,f,e)*transitionRate/2;
       end
       s1 = y(n1);
       s2 = y(n2);
       
       if edgeMap(s1,s2,e,f) > 0
           g(edgeMap(s1,s2,e,f)) = g(edgeMap(s1,s2,e,f)) + Xedge(i,f,e)*transitionRate/2;
       end
   end
end
end