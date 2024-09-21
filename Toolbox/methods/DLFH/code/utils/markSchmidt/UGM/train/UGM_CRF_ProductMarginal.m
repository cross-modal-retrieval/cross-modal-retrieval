function [obj,g] = UGM_CRF_ProductMarginal(w,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct,inferFunc,varargin)
% Objective function given by product of marginals
% (the gradient calculation seems highly-suboptimal)

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
    
    % Compute marginals and logZ
    [nodeBel,edgeBel] = inferFunc(nodePot,edgePot,edgeStruct,varargin{:});
        
    % Update objective
    for n = 1:nNodes
        obj = obj - log(nodeBel(n,Y(i,n)));
    end
    
    % Update gradient
    if nargout > 1
       for n2 = 1:nNodes
           clamped = zeros(nNodes,1);
           clamped(n2) = Y(i,n2);
           [nodeBelConditional,edgeBelConditional] = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clamped,inferFunc,varargin{:});
           for n = 1:nNodes
               for s = 1:nStates(n)
                   for f = 1:nNodeFeatures
                       if nodeMap(n,s,f) > 0
                           g(nodeMap(n,s,f)) = g(nodeMap(n,s,f)) + Xnode(i,f,n)*(nodeBel(n,s) - nodeBelConditional(n,s));
                       end
                   end
               end
           end
           for e = 1:nEdges
               n1 = edgeEnds(e,1);
               n2 = edgeEnds(e,2);
               for s1 = 1:nStates(n1)
                   for s2 = 1:nStates(n2)
                       for f = 1:nEdgeFeatures
                           if edgeMap(s1,s2,e,f) > 0
                               g(edgeMap(s1,s2,e,f)) = g(edgeMap(s1,s2,e,f)) + Xedge(i,f,e)*(edgeBel(s1,s2,e) - edgeBelConditional(s1,s2,e));
                           end
                       end
                   end
               end
           end
       end
    end
end

end