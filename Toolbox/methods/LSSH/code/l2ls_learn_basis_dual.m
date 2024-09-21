function B = l2ls_learn_basis_dual(X, S, l2norm, Binit)
% Learning basis using Lagrange dual (with basis normalization)
%
% This code solves the following problem:
% 
%    minimize_B   0.5*||X - B*S||^2
%    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
% 
% The detail of the algorithm is described in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007
%
% Written by Honglak Lee <hllee@cs.stanford.edu>
% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

L = size(X,1);
N = size(X,2);
M = size(S, 1);


SSt = S*S';
XSt = X*S';

if exist('Binit', 'var')
    dual_lambda = diag(Binit\XSt - SSt);
else
    dual_lambda = 10*abs(rand(M,1)); % any arbitrary initialization should be ok.
end

c = l2norm^2;
trXXt = sum(sum(X.^2));

lb=zeros(size(dual_lambda));
options = optimset('Algorithm','trust-region-reflective','GradObj','on', 'Hessian','on','Display','off');
%  options = optimset('GradObj','on', 'Hessian','on', 'TolFun', 1e-7);
[x, fval, exitflag, output] = fmincon(@(x) fobj_basis_dual(x, SSt, XSt, X, c, trXXt), dual_lambda, [], [], [], [], lb, [], [], options);
% output.iterations
fval_opt = -0.5*N*fval;
dual_lambda= x;

Bt = (SSt+diag(dual_lambda)) \ XSt';
B_dual= Bt';
fobjective_dual = fval_opt;


B= B_dual;
fobjective = fobjective_dual;


return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

