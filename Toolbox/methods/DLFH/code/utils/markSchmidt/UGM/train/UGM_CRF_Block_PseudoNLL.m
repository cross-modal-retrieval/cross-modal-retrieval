function [NLL,g] = UGM_CRF_Block_PseudoNLL(w,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct,blocks,inferFunc,varargin)

[nNodes,maxState] = size(nodeMap);
nNodeFeatures = size(Xnode,2);
nEdgeFeatures = size(Xedge,2);
nEdges = edgeStruct.nEdges;
edgeEnds = edgeStruct.edgeEnds;
V = edgeStruct.V;
E = edgeStruct.E;
nStates = edgeStruct.nStates;

nInstances = size(Y,1);
nBlocks = length(blocks);
NLL = 0;
g = zeros(size(w));

for i = 1:nInstances
    if edgeStruct.useMex
        [nodePot,edgePot] = UGM_CRF_makePotentialsC(w,Xnode,Xedge,nodeMap,edgeMap,nStates,edgeEnds,int32(i));
    else
        [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
    end
    
    for b = 1:nBlocks
        clamped = Y(i,:);
        clamped(blocks{b}) = 0;
        
        [nodeBel,edgeBel,logZ] = UGM_Infer_Conditional(nodePot,edgePot,edgeStruct,clamped,inferFunc,varargin{:});
        
        % Update Objective
        for n = blocks{b}(:)'
            NLL = NLL - log(nodePot(n,Y(i,n)));
            
            for e = UGM_getEdges(n,edgeStruct)
                n1 = edgeEnds(e,1);
                n2 = edgeEnds(e,2);
                
                if n == n1
                    if clamped(n2) ~= 0 % We only want to count edges within the block once
                        NLL = NLL - log(edgePot(Y(i,n1),Y(i,n2),e));
                    end
                else
                    NLL = NLL - log(edgePot(Y(i,n1),Y(i,n2),e));
                end
            end
        end
        NLL = NLL + logZ;
        
        % Update gradient
        if nargout > 1
            for n = blocks{b}(:)'
                for f = 1:nNodeFeatures
                    for s = 1:nStates(n)
                        if nodeMap(n,s,f) > 0
                            if s == Y(i,n)
                                obs = 1;
                            else
                                obs = 0;
                            end
                            g(nodeMap(n,s,f)) = g(nodeMap(n,s,f)) + Xnode(i,f,n)*(nodeBel(n,s) - obs);
                        end
                    end
                end
                
                for e = UGM_getEdges(n,edgeStruct)
                    n1 = edgeEnds(e,1);
                    n2 = edgeEnds(e,2);
                    
                    if n == n1 && clamped(n2) ~= 0
                        for f = 1:nEdgeFeatures
                            for s = 1:nStates(n)
                                if edgeMap(s,Y(i,n2),e,f) > 0
                                    if s == Y(i,n)
                                        obs = 1;
                                    else
                                        obs = 0;
                                    end
                                    g(edgeMap(s,Y(i,n2),e,f)) = g(edgeMap(s,Y(i,n2),e,f)) + Xedge(i,f,e)*(nodeBel(n,s) - obs);
                                end
                            end
                        end
                    elseif n == n2 && clamped(n1) ~= 0
                        for f = 1:nEdgeFeatures
                            for s = 1:nStates(n)
                                if edgeMap(Y(i,n1),s,e,f) > 0
                                    if s == Y(i,n)
                                        obs = 1;
                                    else
                                        obs = 0;
                                    end
                                    g(edgeMap(Y(i,n1),s,e,f)) = g(edgeMap(Y(i,n1),s,e,f)) + Xedge(i,f,e)*(nodeBel(n,s) - obs);
                                end
                            end
                        end
                    elseif n == n1 && clamped(n2) == 0
                        for f = 1:nEdgeFeatures
                            for s1 = 1:nStates(n1)
                                for s2 = 1:nStates(n2)
                                    if edgeMap(s1,s2,e,f) > 0
                                        if s1 == Y(i,n1) && s2 == Y(i,n2)
                                            obs = 1;
                                        else
                                            obs = 0;
                                        end
                                        g(edgeMap(s1,s2,e,f)) = g(edgeMap(s1,s2,e,f)) + Xedge(i,f,e)*(edgeBel(s1,s2,e) - obs);
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end