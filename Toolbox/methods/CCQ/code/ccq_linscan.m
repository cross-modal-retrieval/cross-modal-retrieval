%function ids = ccq_linscan(queryR, dbQ, dbL2, codebook, knn)
function dist = ccq_linscan(queryR, dbQ, dbL2, codebook, knn)

    m = length(codebook);
    Q2B = cell(m, 1);

    for i = 1:m
        Q2B{i} = queryR' * codebook{i};
    end

    dist = 0;

    for i = 1:m
        dist = dist + Q2B{i}(:, dbQ(i, :) + 1);
    end

    dist = bsxfun(@minus, 2 * dist, dbL2);
    %[~, ids] = sort(dist, 2, 'descend');
    %ids = ids(:, 1:knn)';

end
