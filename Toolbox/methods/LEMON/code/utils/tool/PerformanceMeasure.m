function [macroF1, Accuracy] = PerformanceMeasure(Y, Y_hat)
% Y: n*k, ground-truth labelling matrix
% Y_hat: n*k, predicted labelling matrix
% performance metrics: label-based macroF1 & example-based accuracy

% in case the format is not correct
Y(Y<=0) = 0;
Y(Y>0) = 1;
Y_hat(Y_hat<=0) = 0;
Y_hat(Y_hat>0) = 1;

% calculating precision/recall/F1
[n, k] = size(Y);
F1s = zeros(1, k);  precs = zeros(1, k);    recs = zeros(1, k);
TPs = zeros(1, k);  FPs = zeros(1, k);      TNs = zeros(1, k);      FNs = zeros(1, k);
for i = 1 : k
    [tempF, TPs(1, i), FPs(1, i), TNs(1, i), FNs(1, i)] = computeF1_2(Y(:, i), Y_hat(:, i)); 
    
    right = Y(:, i) & Y_hat(:, i);
    TP = sum(right);
    TPFP = sum(Y_hat(:, i));
    TPFN = sum(Y(:, i));
    if TPFP > 0
        precs(1, i) = TP / TPFP;
    else
        precs(1, i) = 0;
    end
    
    if TPFN > 0
        recs(1, i) = TP / TPFN;
    else
        recs(1, i) = 0.5;
    end
    
    if precs(1, i) + recs(1, i) == 0
        F1s(1, i) = 0;
    else
        F1s(1, i) = 2 * precs(1, i) * recs(1, i) / (precs(1, i) + recs(1, i));
    end
end

% calculating macroF1
macroF1 = mean(F1s);

% calculating example-based accuracy 
Accuracy = sum(Y & Y_hat, 2) ./ sum(Y | Y_hat, 2);
Accuracy(isinf(Accuracy)) = 0.5;
Accuracy(isnan(Accuracy)) = 0.5;
Accuracy = mean(Accuracy);
