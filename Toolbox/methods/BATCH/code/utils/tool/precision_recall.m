function [precision, recall] = precision_recall(ids, Lbase, Lquery)

nquery = size(ids, 2);
K = size(ids, 1);
P = zeros(K, nquery);
R = zeros(K, nquery);

for i = 1 : nquery
    label = Lquery(i, :);
    label(label == 0) = -1;
    idx = ids(:, i);
    imatch = sum(bsxfun(@eq, Lbase(idx(1:K), :), label), 2) > 0;
    LK = sum(imatch);
    if LK == 0
        continue;
    end
    Lk = cumsum(imatch);
    P(:, i) = Lk ./ (1:K)';
    R(:, i) = Lk ./ LK;
end
mP = mean(P, 2);
mR = mean(R, 2);
mP = [mP(1); mP];
mR = [0; mR];

recall = (0.0:0.001:max(mR))';
precision = interpolate_pr(mR, mP, recall)';

end


function precision = interpolate_pr(r, p, recs)

n = numel(p);
if (n ~= numel(r))
    error('two first arguments should be of the same size');
end

for j = 1:numel(recs)
    rec = recs(j);
    done = 0;
    for i = 1:n-1
        if (r(i) <= rec && rec <= r(i+1))
            done = 1;
            if (r(i) == r(i+1))
                precision(j) = (p(i) + p(i+1)) / 2;
            else
                precision(j) = p(i) + (rec - r(i)) * (p(i+1) - p(i)) / (r(i+1) - r(i));
            end
            break;
        end
    end
    
    if ~done
        error('not done for %.2f!', rec);
    end
end

end
