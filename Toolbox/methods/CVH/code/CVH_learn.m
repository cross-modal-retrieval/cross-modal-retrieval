function [model, B] = CVH_learn(feaTrain_vis, feaTrain_text, param, maxbits)
    % I=eye(128,10);%2.28
    lambda = param.lambda;
    Cxx = feaTrain_vis' * feaTrain_vis + lambda * eye(size(feaTrain_vis, 2)); %d*d,dΪͼ���ά��
    Cyy = feaTrain_text' * feaTrain_text + lambda * eye(size(feaTrain_text, 2)); %s*s��sΪ�ı���ά��
    Cxy = feaTrain_vis' * feaTrain_text; %d*s
    [Wx, Wy] = CVHtrainCCA(Cxx, Cyy, Cxy, maxbits);
    % size(feaTrain_vis)
    % size(Wx)
    B1 = compress2code(feaTrain_vis', Wx, 'zero', 0); %size(B1)=maxbits*n
    B2 = compress2code(feaTrain_text', Wy, 'zero', 0); %size(B2)=maxbits*n
    B.B1 = B1';
    B.B2 = B2';
    model.Wx = Wx; %size(Wx)=2292*maxbit
    model.Wy = Wy;
    %model.Wy = model.Wy./repmat(sqrt(sum(abs(model.Wy).^2)),sy,1);
end
