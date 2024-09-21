function  [y] = UGM_Decode_ICM(nodeEnergy, edgeEnergy, edgeWeights, edgeStruct,y)
% INPUT
% nodePot(node,class)
% edgePot(class,class,edge) where e is referenced by V,E (must be the same
% between feature engine and inference engine)
%
% OUTPUT
% nodeLabel(node)

if nargin < 6
    [junk y] = min(nodeEnergy,[],2);
end

if edgeStruct.useMex
    y = UGMep_Decode_ICMC(int32(y-1),nodeEnergy,edgeEnergy,edgeWeights,int32(edgeStruct.edgeEnds-1),int32(edgeStruct.V-1),int32(edgeStruct.E-1),int32(edgeStruct.nStates));
else
    
    [nNodes,maxState] = size(nodeEnergy);
    edgeEnds = edgeStruct.edgeEnds;
    V = edgeStruct.V;
    E = edgeStruct.E;
    nStates = edgeStruct.nStates;
    
    done = 0;
    fprintf('ICM iter = %d, energy = %f\n',0,UGMep_Energy(y,nodeEnergy,edgeEnergy,edgeWeights,edgeEnds));
    iter = 1;
    while ~done
        done = 1;
        y2 = y;
        for n = 1:nNodes
            % Compute Node Potential
            energy = nodeEnergy(n,1:nStates(n));
            
            % Find Neighbors
            edges = E(V(n):V(n+1)-1);
            
            % Multiply Edge Potentials
            for e = edges(:)'
                n1 = edgeEnds(e,1);
                n2 = edgeEnds(e,2);
                
                if n == edgeEnds(e,1)
                    energy = energy + edgeWeights(e)*edgeEnergy(1:nStates(n1),y(n2))';
                else
                    energy = energy + edgeWeights(e)*edgeEnergy(y(n1),1:nStates(n2));
                end
            end
            
            % Assign to Maximum State
            [junk newY] = min(energy);
            if newY ~= y(n)
                y(n) = newY;
                done = 0;
            end
        end
        
        fprintf('ICM iter = %d, energy = %f, changes = %d\n',iter,UGMep_Energy(y,nodeEnergy,edgeEnergy,edgeWeights,edgeEnds),sum(y2~=y));
        iter = iter + 1;
    end
end
end
