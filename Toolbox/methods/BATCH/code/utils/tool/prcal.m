function [p, r, map, ph2] = prcal(score,label,labelvalue,groundtruth)

[Nsamples,label_num]=size(label);
p=zeros(1,9);
r=zeros(1,9);
ph2=0;
map=0;
num_truesamples=sum(groundtruth);

[sorted_val, sorted_ind]=sort(score); 
sorted_truefalse=zeros(1,Nsamples);
for i=1:Nsamples
    if(groundtruth(1,sorted_ind(1,i))==1)
        sorted_truefalse(1,i)=1;
    end
end

%p 
M=[5 10 50 100 200 300 400 500 800 1000];
for i_M=1:length(M)          
    p(1,i_M)=sum(sorted_truefalse(1,1:M(i_M)))/M(i_M); 
end

%r
M=[1000 2000 3000 4000 5000 6000 8000 10000 15000 20000];
for i_M=1:length(M)          
    r(1,i_M)=sum(sorted_truefalse(1,1:M(i_M)))/num_truesamples;
end

%hd2
hd2_ind=find(score<=2);
hd2_length=length(hd2_ind);
if isempty(hd2_ind)
    ph2=0;
else
    ph2=sum(sorted_truefalse(1,1:hd2_length))/hd2_length;
end


%map200
right_num=0;
right_sum=0;
for i=1:200
    if(sorted_truefalse(1,i)==1)
        right_num=right_num+1;
        right_sum=right_sum+sum(sorted_truefalse(1,1:i))/i;
    end
end
if(right_num~=0)
    map=right_sum/right_num;
end
%save m sorted_truefalse right_sum right_num sorted_val;
