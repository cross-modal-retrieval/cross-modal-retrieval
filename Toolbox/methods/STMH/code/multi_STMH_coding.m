function outG0 = multi_STMH_coding(inXCell, F, numzeros)
    % input:
    %       inXcell: the size of inXcell is d by n
    %       F: cluster centers(d by c)
    %       numzeros: number of zeros in each row of outG0
    % output:
    %       outG0: the output cluster indicator (n by c)
    %
    % Fix inXcell, D, F, update G0
    n = size(inXCell, 2);
    c = size(F, 2);
    outG0 = zeros(n, c);

    for i = 1:n
        xVec = inXCell(:, i);
        outG0(i, :) = searchBestIndicator(xVec, F, numzeros);
    end

end

%% function searchBestIndicator
function outVec = searchBestIndicator(xCell, F, numzeros)

    c = size(F, 2);
    tmp = zeros(c, 1);
    obj = zeros(c, 1);

    for j = 1:c
        obj(j, 1) = obj(j, 1) + norm(xCell - F(:, j))^2;
    end

    [~, idx] = sort(obj, 'ascend');
    tmp(idx(1:numzeros)) = 1;
    outVec = tmp;
end
