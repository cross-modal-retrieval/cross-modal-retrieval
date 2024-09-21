function [S,Xanchor,Yanchor]=constructS(XTrain,YTrain,Label)

    [Xn,Xdim] = size(XTrain);
    [Yn,Ydim] = size(YTrain);
%     Xindex=randperm(Xn);
%     Yindex=randperm(Yn);
%     Xanchor=XTrain(Xindex(1:100),:);%�ر��ر�ע������ط�
%     Yanchor=YTrain(Xindex(1:100),:);%�ر��ر�ע������ط�
%     [XIdx,Xanchor]=Kmeans(XTrain,40,'dist','sqEuclidean','rep',20);
%      [YIdx,Yanchor]=Kmeans(YTrain,40,'dist','sqEuclidean','rep',20);
%     load Xanchor_50;
%     load Yanchor_50;
    Xanchor=construct_Anchor(XTrain,Label);
    Yanchor=construct_Anchor(YTrain,Label);
    %% get Z
    XZ = zeros(Xn,50); %ê����ĿΪ300
    YZ = zeros(Yn,50); %ê����ĿΪ300
    XDis = sqdist(XTrain',Xanchor');
    YDis = sqdist(YTrain',Yanchor');
%     clear XTrain;
%     clear XAnchor;
    Sx=5;%ע������ط�  5
    Sy=5;
    Xsigma=0;
    Ysigma=0;
    Xval = zeros(Xn,Sx);
    Yval = zeros(Yn,Sy);
    Xpos = Xval;
    Ypos = Yval;
    for i = 1:Sx
        [Xval(:,i),Xpos(:,i)] = min(XDis,[],2);
        Xtep = (Xpos(:,i)-1)*Xn+[1:Xn]';
        XDis(Xtep) = 1e60; 
    end
       for i = 1:Sy
        [Yval(:,i),Ypos(:,i)] = min(YDis,[],2);
        Ytep = (Ypos(:,i)-1)*Yn+[1:Yn]';
        YDis(Ytep) = 1e60;
       end
    if Xsigma == 0
       Xsigma = mean(Xval(:,Sx).^0.5);
    end
    if Ysigma == 0
       Ysigma = mean(Yval(:,Sy).^0.5);
    end
    Xval = exp(-Xval/(1/1*Xsigma^2));
    Yval = exp(-Yval/(1/1*Ysigma^2));
%     Xval = repmat(sum(Xval,2).^-1,1,Sx).*Xval; %% normalize
%     Yval = repmat(sum(Yval,2).^-1,1,Sy).*Yval; %% normalize
    Xtep = (Xpos-1)*Xn+repmat([1:Xn]',1,Sx);
    Ytep = (Ypos-1)*Yn+repmat([1:Yn]',1,Sy);
    XZ([Xtep]) = [Xval];
    YZ([Ytep]) = [Yval];

    
    %% ��һ�ְ汾 Sx=3   Sy=5 ����ȽϺ�
    XZ=double(XZ>0);
    YZ=double(YZ>0);
    XLength = sqrt(sum(XZ.^2, 2));
    XLength(XLength == 0) = 1e-8; % avoid division by zero problem for unlabeled rows
    XLambda = 1 ./ XLength;
    XL = diag(sparse(XLambda)) * XZ;
    XS=XL*XL';
    YLength = sqrt(sum(YZ.^2, 2));
    YLength(YLength == 0) = 1e-8; % avoid division by zero problem for unlabeled rows
    YLambda = 1 ./ YLength;
    YL = diag(sparse(YLambda)) * YZ;
    YS=YL*YL';
    S=0.045*XS+0.955*YS;
 
end