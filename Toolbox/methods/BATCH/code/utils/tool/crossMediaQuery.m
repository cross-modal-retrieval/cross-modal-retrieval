function [ maps ] = crossMediaQuery( cat_te, cat_tr, X2_query_binary, X1_binary )
[numquery dimquery] = size(X2_query_binary);
fprintf('query size: %d*%d.\n', numquery, dimquery);
[numbase dimbase] = size(X1_binary);
fprintf('base size: %d*%d.\n', numbase, dimbase);
maps = zeros(numquery , 1);

% %% compute the number of true neighbors in ground truth
% nb_num = zeros(10,1);
% for i=1:10
%     ind = find(cat_tr==i);
%     nb_num(i) = size(ind , 1);
% end

%% calculate mAP for every query
R = 50; 
for i=1:numquery
    query = X2_query_binary(i,:);
    query_cat = cat_te(i);
    query_mat = repmat(query, numbase, 1);
    diff = xor(query_mat, X1_binary);
    sum_diff = sum(diff, 2);
    [B,IX] = sort(sum_diff, 'ascend');
    
    temp = 0;
    right = 0;
    for j=1:R
       ind = IX(j);
       if query_cat == cat_tr(ind)
           right = right + 1;
           precision = right/j;
           temp = temp + precision;
       end
    end
    if right~=0
        maps(i) = temp/right;
    end
end

end

