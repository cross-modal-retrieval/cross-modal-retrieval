function [f,g,H] = fobj_basis_dual(dual_lambda, SSt, XSt, X, c, trXXt)
% Compute the objective function value at x
L= size(XSt,1);
M= length(dual_lambda);

SSt_inv = inv(SSt + diag(dual_lambda));

% trXXt = sum(sum(X.^2));
if L>M
    % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
    f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
    
else
    % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
    f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
end
f= -f;

if nargout > 1   % fun called with two output arguments
    % Gradient of the function evaluated at x
    g = zeros(M,1);
    temp = XSt*SSt_inv;
    g = sum(temp.^2) - c;
    g= -g;
    
    
    if nargout > 2
        % Hessian evaluated at x
        % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
        H = -2.*((temp'*temp).*SSt_inv);
        H = -H;
    end
end

return