
function labels = lclassify(TEST, TRAIN, train_lab);

% Least squares classifier.
%
%   TEST - NxK real matrix
%   TRAIN - LxK real matrix
%   train_lab - Lx1 vector with integer values [1..m]
%
% Returns:
% 
%   labels - Nx1 vector with integer values [1..m]
%
% Primary purpose here:
%
%   Can be used for partially labeled learning (or regression with
%   minor modifications). In that setting 
%   the TEST and TRAIN are eigenvectors of the Laplacian matrix
%   of the adjacency graph for the data. TEST corresponds to
%   unlabeled points and TRAIN corresponds to labeled points.
%   train_lab contains known labels for the labeled part of the data.
%   Output labels contains labelling of the unlabeled data points.
%
% For a detailed description of the algorithm please refer to 
% University of Chicago
% Computer Science Technical Report TR-2002-12
% Mikhail Belkin, Partha Niyogi
% Semi-supervised learning on manifolds
% http://www.cs.uchicago.edu/research/publications/techreports/TR-2002-12
%
%
% Author: 
%
%   Mikhail Belkin 
%   misha@math.uchicago.edu
%

if (nargin < 3)
  disp(sprintf('ERROR: Too few arguments given.\n'));
  disp(sprintf('USAGE:\nlabels = lclassify(TEST, TRAIN, train_lab); '));  
end;  

mx = max(train_lab)
mn = min(train_lab)
classes = mx - mn + 1

vote = zeros(size(TEST,1),classes);
conf = zeros(size(TEST,1),classes);
zz = zeros(size(TEST,2),classes);
tl = train_lab - mn;

for i=0:classes-1

  l=tl*0 - 1;;
  l(find(tl~=i))=1;
  
  if ( size (find(tl==i),2) == 0) 
    disp ('Warning: no elements');
  end;
  
  T = (TRAIN'*TRAIN)^(-1);
   
  %zz =  (train'*train)^(-1)*train'*l;
  zz(:,i+1) = T * TRAIN' * l;
  
  %l = 0*test_l;
  v = test*zz(:,i+1);
  
  vote(find(v <=0),i+1) = 1;
  conf(:,i+1) = v;
%   l(find(test*zz <=0)) = 0;

end;

[Z,I] = sort(conf,2);

labels = I(:,1)-1+mn;


