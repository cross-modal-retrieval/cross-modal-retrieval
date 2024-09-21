function [edges,adj] = UGMlearn2_chowLiu(X,options)
% Chow-Liu w/ tabular CPDs

if nargin < 2 
    options = [];
end

[alpha] = myProcessOptions(options,'alpha',eps);

[nSamples,nNodes] = size(X);
nStates = max(X(:));

%% Compute sufficient statistics of data
for n = 1:nNodes
    for s = 1:nStates
        p1(n,s) = sum(X(:,n)==s);
    end
end
for n1 = 1:nNodes
    for n2 = 1:nNodes
        for s1 = 1:nStates
            for s2 = 1:nStates
                p2(n1,n2,s1,s2) = sum(X(:,n1)==s1 & X(:,n2)==s2);
            end
        end
    end
end

%% Estimate all unconditional parameters
for n = 1:nNodes
    for s = 1:nStates
        w1(n,s) = p1(n,s)+alpha;
    end
    w1(n,:) = w1(n,:)/sum(w1(n,:));
end

%% Estimate all 1-parent conditional parameters
for n1 = 1:nNodes
    for n2 = 1:nNodes
        for s2 = 1:nStates
            for s1 = 1:nStates
                w2(n1,n2,s1,s2) = p2(n1,n2,s1,s2)+alpha;
            end
            w2(n1,n2,:,s2) = w2(n1,n2,:,s2)/sum(w2(n1,n2,:,s2));
        end
    end
end

%% Compute unconditional log-likelihoods
logp1 = zeros(nNodes,1);
for n = 1:nNodes
    for s = 1:nStates
        logp1(n) = logp1(n) + p1(n,s)*log(w1(n,s));
    end
end

%% Compute conditional log-likelihoods
logp2 = zeros(nNodes);
for n1 = 1:nNodes
    for n2 = 1:nNodes
        for s1 = 1:nStates
            for s2 = 1:nStates
                logp2(n1,n2) = logp2(n1,n2) + p2(n1,n2,s1,s2)*log(w2(n1,n2,s1,s2));
            end
        end
    end
end

if 0 % Show weight matrix
    for n1 = 1:nNodes
        for n2 = 1:nNodes
            weights(n1,n2) = logp2(n1,n2) - logp1(n1);
        end
    end
    weights
    max(max(abs(weights-weights')))
    pause
end


%% Weights are unconditionals-conditionals
edgeEnds = zeros(0,3);
for n1 = 1:nNodes
    for n2 = n1+1:nNodes
        edgeEnds(end+1,:) = [n1 n2 logp1(n1) - logp2(n1,n2)];
    end
end

%% Solve
E = minSpan(nNodes,edgeEnds);

%% Make set of selected edges
edges = zeros(sum(E),2);
e2 = 1;
for e = 1:length(E)
    if E(e)==1
        edges(e2,:) = edgeEnds(e,1:2);
        e2 = e2+1;
    end
end

%% Make adjacency matrix
if nargout > 1
adj = zeros(nNodes);
for e = 1:length(E)
   if E(e)==1
      adj(edgeEnds(e,1),edgeEnds(e,2)) = 1; 
   end
end
adj = adj+adj';
end