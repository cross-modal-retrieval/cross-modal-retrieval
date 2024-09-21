function [L Ln A An] = Laplacian_GK(X, para)
% each column is a data

if isfield(para, 'k')
    k = para.k;
else
    k = 15;
end;
if isfield(para, 'sigma')
    sigma = para.sigma;
else
    sigma = 1;
end;

	[nFea, nSmp] = size(X);
    D = L2_distance_1(X,X);
    W = spalloc(nSmp,nSmp,20*nSmp);

    [dumb idx] = sort(D, 2); % sort each row

    for i = 1 : nSmp
    W(i,idx(i,2:k+1)) = 1;
    %    W(i,idx(i,2:k+1)) = exp(-D(i,idx(i,2:k+1))/sigma^2);         
    end
    W = (W+W')/2;
    
    D = diag(sum(W,2));
    L = D - W;
    A = W;
    
    Dd = diag(D)+10^-12;
    Dn=diag(sqrt(1./Dd)); Dn = sparse(Dn);
    An = Dn*W*Dn; An = (An+An')/2;
    Ln=speye(size(W,1)) - An;