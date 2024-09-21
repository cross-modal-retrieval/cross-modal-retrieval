function precision = precision_at_k(ids, Lbase, Lquery)

nquery = size(ids, 2);
K = 1000;
P = zeros(K, nquery);

for i = 1 : nquery
    label = Lquery(i, :);
    label(label == 0) = -1;
    idx = ids(:, i);
    imatch = sum(bsxfun(@eq, Lbase(idx(1:K), :), label), 2) > 0;
    Lk = cumsum(imatch);
    P(:, i) = Lk ./ (1:K)';
end
precision = mean(P, 2);

end
