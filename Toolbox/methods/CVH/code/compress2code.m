%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Distributed under GNU General Public License (see license.txt for details).
%
% Copyright (c) 2012 Linus ZHEN Yi
% All Rights Reserved.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function compress data to hash codes
% input:
%   data : data matrix (Dim x N), each point is a column
%   W : projection matrix [Dim x Codelength]
%   threshname : name of thresholding method
%   T : threshold vector [Codelength x 1] to be used
% output:
%   code : M x N matrix (each column is a point compacted in words)
%   T : threshold vector [Codelength x 1] used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [code,T] = compress2code(data, W, threshname, T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % get projected data matrix
    U = W'*data; % codelength x N, each point is a column
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % choose threshold
    if nargin < 4
        switch threshname
            case 'zero'
                T = 0;
            case 'mean'
                T = mean(W'*data,2);
            case 'median'
                T = median(W'*data,2);
            case 'quater-mean'
                quatervec = range(U,2)/4;
                minvec = min(U,[],2);
                maxvec = max(U,[],2);
                T.low = double(bsxfun(@plus,minvec, quatervec));
                T.high = double(bsxfun(@minus,maxvec, quatervec));
            case 'quater-median'
                Us = sort(U,2,'ascend');
                T.low = double(Us(:,floor(size(data,2)/4)));
                T.high = double(Us(:,floor(size(data,2)*3/4)));
            case 'learn'
                nBin = 20;
                kappa = [1,1,1];
                T = zeros(size(U,1),1);
                for m = 1:size(U,1)
                    [nP, cP] = hist(U(m,:),nBin);
                    leftnP = zeros(nBin,1);
                    rightnP = zeros(nBin,1);
                    for npos = 1:nBin
                        leftnP(npos) = sum(nP(1:npos-1));
                        rightnP(npos) = sum(nP(npos+1:end));
                    end
                    costP = kappa(1)*leftnP.^2+kappa(2)*rightnP.^2+kappa(3)*nP'*size(U,2);
                    [~, i] = min(costP);
                    T(m) = cP(i);
                end
            otherwise
                error('The threshold method is not supported');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % apply threshold
    if strcmp(threshname,'quater-median') || strcmp(threshname,'quater-mean') 
        Ulow = bsxfun(@lt,U,T.low);
        Uhigh = bsxfun(@gt,U,T.high);
        U1 = double(Ulow | Uhigh);
        U2 = double(bsxfun(@gt,U,0));
        r = floor(size(W,2)/2);
        U = [U1(1:r,:);U2(1:r,:)];
    elseif strcmp(threshname,'learn')% one value generate two bits
        U1 = double(bsxfun(@gt,U,0));
        U2 = double(bsxfun(@gt,U,T));
        r = floor(size(W,2)/2);
        U = [U1(1:r,:);U2(1:r,:)];
    else % normal threshold
        U = double(bsxfun(@gt,U,T));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   code = U;
end
