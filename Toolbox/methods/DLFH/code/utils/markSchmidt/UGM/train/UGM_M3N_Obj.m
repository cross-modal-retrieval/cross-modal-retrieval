function [obj,sg] = UGM_M3N_Obj(w,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct,decodeFunc,varargin)
% UGM_M3N_Obj(w,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct,inferFunc,varargin)

[nNodes,maxState] = size(nodeMap);
nNodeFeatures = size(Xnode,2);
nEdgeFeatures = size(Xedge,2);
nEdges = edgeStruct.nEdges;
edgeEnds = edgeStruct.edgeEnds;
nStates = edgeStruct.nStates;

nInstances = size(Y,1);
obj = 0;
sg = zeros(size(w));

for i = 1:nInstances
    
    % Make potentials
    if edgeStruct.useMex
        [nodePot,edgePot] = UGM_CRF_makePotentialsC(w,Xnode,Xedge,nodeMap,edgeMap,nStates,edgeEnds,int32(i));
    else
        [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,i);
    end
    
    % Update based on true label
    if edgeStruct.useMex
        obj = obj - UGM_LogConfigurationPotentialC(Y(i,:),nodePot,edgePot,edgeEnds);
    else
        obj = obj - UGM_LogConfigurationPotential(Y(i,:),nodePot,edgePot,edgeEnds);
    end
    
    % Make Loss-Augmented Potentials (assumes Hamming loss)
    nodePot = nodePot*exp(1);
    for n = 1:nNodes
        nodePot(n,Y(i,n)) = nodePot(n,Y(i,n))/exp(1);
    end
    
    if ~isLegal(nodePot) || ~isLegal(edgePot)
        fprintf('Objective over-flowed or under-flowed\n');
        obj = inf; % Over-flow or under-flow
        return
    end
    
    % Do Loss-Augmented Decoding
    yMAP = decodeFunc(nodePot,edgePot,edgeStruct,varargin{:});
    
    % Update objective
    if edgeStruct.useMex
        obj = obj + UGM_LogConfigurationPotentialC(yMAP,nodePot,edgePot,edgeEnds);
    else
        obj = obj + UGM_LogConfigurationPotential(yMAP,nodePot,edgePot,edgeEnds);
    end
    
    % Update subgradient
    if edgeStruct.useMex
        % Updates in-place
        nodeBel = zeros(size(nodePot));
        edgeBel = zeros(size(edgePot));
        
        for n = 1:nNodes
            nodeBel(n,yMAP(n)) = 1;
        end
        for e = 1:nEdges
            n1 = edgeEnds(e,1);
            n2 = edgeEnds(e,2);
            edgeBel(yMAP(n1),yMAP(n2),e) = 1;
        end
        
        UGM_CRF_NLLC(sg,int32(i),nodeBel,edgeBel,edgeEnds,nStates,nodeMap,edgeMap,Xnode,Xedge,Y);
    else        
        if nargout > 1
            for n = 1:nNodes
                for s = 1:nStates(n)
                    for f = 1:nNodeFeatures
                        if nodeMap(n,s,f) > 0
                            if s == Y(i,n)
                                O = 1;
                            else
                                O = 0;
                            end
                            if s == yMAP(n)
                                E = 1;
                            else
                                E = 0;
                            end
                            sg(nodeMap(n,s,f)) = sg(nodeMap(n,s,f)) + Xnode(i,f,n)*(E - O);
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
                                if s1 == Y(i,n1) && s2 == Y(i,n2)
                                    O = 1;
                                else
                                    O = 0;
                                end
                                if s1 == yMAP(n1) && s2 == yMAP(n2)
                                    E = 1;
                                else
                                    E = 0;
                                end
                                sg(edgeMap(s1,s2,e,f)) = sg(edgeMap(s1,s2,e,f)) + Xedge(i,f,e)*(E - O);
                            end
                        end
                    end
                end
            end
        end
    end
end