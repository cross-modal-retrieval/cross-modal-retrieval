function [G, F, B] = DCH_discrete(X,T, y,B,gmap,Fmap,tol,maxItr,debug)

% ---------- Argument defaults ----------
if ~exist('debug','var') || isempty(debug)
    debug=1;
end
if ~exist('tol','var') || isempty(tol)
    tol=1e-5;
end
if ~exist('maxItr','var') || isempty(maxItr)
    maxItr=1000;
end
nu = Fmap.nu;
mu = Fmap.mu;
delta = 1/nu;
deltat = 1/mu;
% ---------- End ----------

% label matrix N x c
if isvector(y) 
    Y = sparse(1:length(y), double(y), 1); Y = full(Y);
else
    Y = y;
end


% G-step
switch gmap.loss
    case 'L2'
        [Wg, ~, ~] = RRC(B, Y, gmap.lambda); % (Z'*Z + gmap.lambda*eye(nbits))\Z'*Y;
    case 'Hinge'
        svm_option = ['-q -s 4 -c ', num2str(1/gmap.lambda)];
        model = train(double(y),sparse(B),svm_option);
        Wg = model.w';
end
G.W = Wg;

%F-step
[WF, ~, ~] = RRC(X, B, Fmap.lambda);
[WFt, ~, ~] = RRC(T, B, Fmap.lambda);

F.W = WF; F.nu = nu;
F.Wt = WFt; F.mu = mu;


i = 0; 
while i < maxItr  
    i=i+1;  
    
    if debug,fprintf('Iteration  %03d: ',i);end
    
    % B-step
  
%         XF = X*WF;
%         TF = T*WFt;
   
    switch gmap.loss
        case 'L2'
%             Q = Y*Wg';
%             B = zeros(size(B));          
%             for time = 1:10           
%                Z0 = B;
%                 for k = 1 : size(B,2)
%                     Zk = B; Zk(:,k) = [];
%                     Wkk = Wg(k,:); Wk = Wg; Wk(k,:) = [];                    
%                     B(:,k) = sign(Q(:,k) -  Zk*Wk*Wkk');
%                 end
%                 
%                 if norm(B-Z0,'fro') < 1e-6 * norm(Z0,'fro')
%                     break
%                 end
%                 B = sign(B);
% %                 errors = norm(Y - B*Wg, 'fro') + gmap.lambda*norm(Wg, 'fro');
% %                 fprintf('  erros=%g \n', errors);
%             end

%              % update B without descrete constraints
            B = (Wg*Wg' + Fmap.lambda*eye(size(Wg, 1))) \ ( Wg*Y');
            B = B';
            B = sign(B);
%             errors = norm(Y - B*Wg, 'fro') + gmap.lambda*norm(Wg, 'fro');
%             fprintf('  erros=%g \n', errors);
        case 'Hinge' 
            
            for ix_z = 1 : size(B,1)
                w_ix_z = bsxfun(@minus, Wg(:,y(ix_z)), Wg);
                B(ix_z,:) = sign(2*nu*XF(ix_z,:) + delta*sum(w_ix_z,2)');
            end
        case 'Quant'
             % update B without descrete constraints
            B = (Wg*Wg' + Fmap.lambda*eye(size(Wg, 1))) \ ( Wg*Y');
            B = B';
            
            errors = norm(Y - B*Wg, 'fro') + gmap.lambda*norm(Wg, 'fro');
            fprintf('  erros=%g \n', errors);
    end

    
    % G-step
    switch gmap.loss
    case 'L2'
        [Wg, ~, ~] = RRC(B, Y, gmap.lambda); % (Z'*Z + gmap.lambda*eye(nbits))\Z'*Y;
    case 'Hinge'        
        model = train(double(y),sparse(B),svm_option);
        Wg = model.w';
    end
    G.W = Wg;
%     
%     F-step 
    WF0 = WF;
    WFt0 = WFt;
    
    [WF, ~, ~] = RRC(X, B, Fmap.lambda);
    [WFt, ~, ~] = RRC(T, B, Fmap.lambda);
   
    F.W = WF; F.nu = nu;
    F.Wt = WFt; F.mu = mu;
    
    
    errors = norm(Y - B*Wg, 'fro');
    errors2 = norm(Wg, 'fro');
%     bias = norm(B-X*WF,'fro');
%     biast = norm(B-T*WFt, 'fro');
    
    
    if debug
%         total(i) = errors + nu*bias + mu*biast;
%         fprintf('  erros=%g , bias=%g, biast=%g, total=%g \n',errors, nu*bias, mu*biast, total(i)); 
        total(i) = errors + Fmap.lambda * errors2;
        fprintf('  erros=%g \n', total(i)); 
    end
    
%     if bias < tol*norm(B,'fro') && biast < tol*norm(B,'fro')
%             break;
%     end 
    
    
%     if norm(WF-WF0,'fro') < tol * norm(WF0) && norm(WFt-WFt0,'fro') < tol * norm(WFt0)
%         break;
%     end
end

% fprintf('finished 20 iterations! \n');

end