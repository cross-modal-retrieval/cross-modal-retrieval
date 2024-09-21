function [trainCode, obj] = minKLD(alpha, P, bit, initC)
% minKLD:   minimizing KL-divergence, developed from tsne_p.m of t-SNE by Laurens van der Maaten
% Input: 
%   alpha: model parameter \alpha (1 x 1)
%   P: probability distribution before hashing (n x n)
%   bit: length of hash codes (1 x 1)
%   initC: initial values of to-be-learnt hash code matrix (n x bit)
% Output:
%   trainCode: learnt hash code matrix of training data (n x bit)
%   obj: optimal value of the objective function

    % Initialize some variables
    n = size(P, 1);                                    % number of instances
    momentum = 0.5;                                    % initial momentum
    max_iter = 100;                                    % maximum number of iterations
    epsilon = 500;                                     % initial learning rate
    min_gain = .01;                                    % minimum gain for delta-bar-delta
    
    % Make sure P-vals are set properly
    P(1:n + 1:end) = 0;                                 % set diagonal to zero
    P = 0.5 * (P + P');                                 % symmetrize P-values
    P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
    const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
        
    % Initialize the solution
    if nargin < 4
        trainCode = .0001 * randn(n, bit); 
    else
        trainCode = initC;
    end
    y_incs  = zeros(size(trainCode));
    gains = ones(size(trainCode));
    
    % Record the minimal objective function and the corresponding optimal
    % hash code matrix of training data
    minCost = realmax;
    minTrainCode = trainCode;
    
    % Run the iterations
    for iter=1:max_iter        
        % Compute joint probability that point i and j are neighbors
        sum_trainCode = sum(trainCode .^ 2, 2);
        % Hamming distance --> t-distribution
        num = 1 ./ (1 + 0.25 * bsxfun(@plus, sum_trainCode, bsxfun(@plus, sum_trainCode', -2 * (trainCode * trainCode')))); 
        num(1:n+1:end) = 0;                                                 % set diagonal to zero
        Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
        
        % Compute the gradients (faster implementation)
        L = (P - Q) .* num;
        y_grads = (diag(sum(L, 2)) - L) * trainCode + 2 * alpha * (abs(trainCode) - 1) .* sign(trainCode);
            
        % Update the solution
        gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
              + (gains * .8) .* (sign(y_grads) == sign(y_incs));
        gains(gains < min_gain) = min_gain;
        y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
        trainCode = trainCode + y_incs;
        % Keep zero mean values
        trainCode = bsxfun(@minus, trainCode, mean(trainCode, 1));     
        
        % Keep the minimal objective function value obtained until now
        cost = const - sum(P(:) .* log(Q(:))) + alpha * norm(abs(trainCode) - 1, 'fro')^2;
        if cost < minCost
            minCost = cost;
            minTrainCode = trainCode;
        end
        
        % Print out progress
        %if ~rem(iter, 10)
        %    %disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
        %end
    end
    
    trainCode = minTrainCode;
    obj = minCost;
