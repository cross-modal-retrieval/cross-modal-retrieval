function [ recall,precision ] = crossMediaQuery_pr( cat_te, cat_tr, X2_query_binary, X1_binary )
[numquery dimquery] = size(X2_query_binary);
% fprintf('query size: %d*%d.\n', numquery, dimquery);
[numbase dimbase] = size(X1_binary);
% fprintf('base size: %d*%d.\n', numbase, dimbase);

%% compute the number of true neighbors in ground truth
nb_num = zeros(10,1);
% for i=1:10
%     ind = find(cat_tr==i);
%     nb_num(i) = size(ind , 1);
% end

point_plot_perc = 0.1:0.02:1;
plot_len = length(point_plot_perc);
precision = zeros(numquery , plot_len);
recall = point_plot_perc;

%% calculate precision and recall for every query
for i=1:numquery
    query = X2_query_binary(i,:);
    query_cat = cat_te(i,:);
%     point_plot = point_plot_perc * nb_num(query_cat);
    num_similar = sum(query_cat * cat_tr' > 0);
    point_plot = point_plot_perc * (num_similar);
    
    query_mat = repmat(query, numbase, 1);
    diff = xor(query_mat, X1_binary);
    sum_diff = sum(diff, 2);
    [B,IX] = sort(sum_diff, 'ascend');
    
    point_plot_ind = 1;
    right = 0;
    for j=1:numbase
        ind = IX(j);
        if (query_cat * cat_tr(ind,:)')>0
            right = right + 1;
            precision_temp = right/j;
            if right >= point_plot(point_plot_ind)
                precision(i, point_plot_ind) = precision_temp;
                point_plot_ind = point_plot_ind + 1;
            end
        end
    end
end
precision = mean(precision);
end

