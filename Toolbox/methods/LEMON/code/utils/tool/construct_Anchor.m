function Anchor=construct_Anchor(Train,Test)
    Center=zeros(10,size(Train,2));
    for i=1:10
        Index=find(Test(:,i)==1);
        Center(i,:)=mean(Train(Index,:));
    end
    Random1=0.01*randn(10,size(Train,2));
    Random2=0.02*randn(10,size(Train,2));
    Anchor=[Center;Center+Random1;Center-Random1;Center+Random2;Center-Random2];
end