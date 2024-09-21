function idx = ccq_nn(Cx, X, Cy, Y, lambda)

if nargin == 2
    dist = sqdist(Cx, X);
    [~, idx] = min(dist); 
elseif nargin == 5
    dist = sqdist(Cx, X) + lambda * sqdist(Cy, Y);
    [~, idx] = min(dist);
else
    error('Wrong number of input arguments.');
end

end
