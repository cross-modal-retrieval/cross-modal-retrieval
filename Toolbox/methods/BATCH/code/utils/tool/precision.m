function p=precision(score,D_truth,labelvalue,groundtruth,K)
    
    if ~exist('K','var')
        K = size(D_truth,1);
    end
    [Nsamples,label_num]=size(D_truth);
    [sorted_val, sorted_ind]=sort(score); 
    sorted_truefalse=zeros(1,Nsamples);
    for i=1:Nsamples
        if(groundtruth(1,sorted_ind(1,i))==1)
            sorted_truefalse(1,i)=1;
        end
    end

%     for i_M=1:length(M)          
%         p(1,i_M)=sum(sorted_truefalse(1,1:M(i_M)))/M(i_M); 
%     end

    p=cumsum(sorted_truefalse(1,1:K))' ./ (1:K)';
end