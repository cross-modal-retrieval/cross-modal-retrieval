function topN= calTop_k(hammTrainTest,feaTrain_class,feaTest_class,nbits) %µ•±Í«©
fprintf('compute the performance\n');
% nbits = 64;
disp(nbits);
%cbTrain = compactbit(codeTrain(:,1:nbits));
%cbTest  = compactbit(codeTest(:,1:nbits));
%hammTrainTest  = hammingDist(cbTest,cbTrain)';
database_label = feaTrain_class;
[sort_val, sort_idx]= sort(hammTrainTest, 1, 'ascend');
for i = 1:size(hammTrainTest, 2)
    qry_label = feaTest_class(i) ;
    ret_label = database_label(sort_idx(:,i));
    for j = 100:100:2000
        precision(i, j/100) = length(find(ret_label(1:j)==qry_label))/j ;
    end
end
topN = mean(precision);
end

