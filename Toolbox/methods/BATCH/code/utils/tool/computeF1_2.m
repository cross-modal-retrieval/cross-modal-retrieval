function [F1, TP, FP, TN, FN] = computeF1_2(Y, Y_hat)
% Y: n*1
% Y_hat: n*1

% in case the format is not correct
Y(Y<=0) = -1;
Y(Y>0) = 1;
Y_hat(Y_hat<=0) = -1;
Y_hat(Y_hat>0) = 1;

% start
n = size(Y,1);
TP = nnz(Y_hat==1&Y==1);
FP = nnz(Y_hat==1&Y==-1);
FN = nnz(Y_hat==-1&Y==1);
TN = nnz(Y_hat==-1&Y==-1);

if n~=(TP+FP+FN+TN)
    disp('n~=(TP+FP+FN+TN)!');
    F1 = 0;
else
    if TP+FP == 0
        prec = 0;
    else
        prec = TP/(TP+FP);
    end
    if TP + FN == 0
        rec = 0.5;
    else
        rec = TP/(TP+FN);
    end
    if prec+rec == 0
        F1 = 0;
    else
        F1 = 2*prec*rec / (prec+rec);
    end
    %fprintf(1, 'F1 = %d ---- TP:%d, FP:%d, FN:%d, TN:%d, Prec:%d, rec:%d\n', F1, TP, FP, FN, TN, prec, rec);
end