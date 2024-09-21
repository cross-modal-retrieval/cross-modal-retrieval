function [model, B] = IMH_learn(feaTrain_vis, feaTrain_text, IMHparam, maxbits)
    %% ���ò���
    fprintf('parameter setting\n');
    beta = IMHparam.beta;
    lambda = IMHparam.lambda;
    para.k = IMHparam.k;
    para.sigma = IMHparam.sigma;

    X1 = feaTrain_vis';
    X2 = feaTrain_text';
    [m1, imgNum] = size(X1);
    [m2, textNum] = size(X2);
    n1 = imgNum; %��n2��Ϊn1
    n2 = textNum; %������n2
    %% Define the related matrix for inter-modality
    tmp1 = eye(n1); %diag(ones(1,n1)); %����һ���Խ���Ԫ��Ϊ1��n1��n1�ľ���n1Ϊͼ��ѵ�������������
    S1 = tmp1(1:n1, :); %����ѡ����󡪡��Լ���n1�޸�Ϊm1
    clear tmp1;
    tmp2 = diag(ones(1, n2)); %tmp2�ĳɴ�СΪn2��n2�ľ���n2Ϊ�ı�ѵ�������������
    S2 = tmp2(1:n1, :);
    clear tmp2;
    U = 10000000000 * diag(ones(1, n1)); %���Խ����ϵ�Ԫ����1��Ϊ10000,U in the paper
    %% Steps used to get M and B
    eyemat_1 = eye(imgNum); %��S1�ȼ�
    eyemat_2 = eye(textNum); %��S2�ȼ�
    eyemat_m_1 = eye(m1); %m1Ϊͼ��ѵ������ά��
    eyemat_m_2 = eye(m2);
    M1 = X1 * X1' + beta * eyemat_m_1; %M1�Ĵ�СΪͼ��ѵ������ά��
    M2 = X2 * X2' + beta * eyemat_m_2; %M2�Ĵ�СΪ�ı�ѵ������ά��
    A1 = eyemat_1 - (X1') / M1 * X1; %B in the paper
    A2 = eyemat_2 - (X2') / M2 * X2;

    %% Steps used to get L1 and L2
    [L, Ln] = Laplacian_GK(X1, para);
    L1 = L;
    [L, Ln] = Laplacian_GK(X2, para);
    L2 = L;
    %% Get Y1 (v, eigval)
    % D = lambda*eyemat-lambda*lambda*eyemat/(L1+lambda*eyemat) + lambda*eyemat-lambda*lambda*eyemat/(L2+lambda*eyemat) + alpha*(Lc-Lc*X'/(X*Lc*X'+beta*eyemat_m)*X*Lc);
    C2 = (A2 + lambda * L2 + S2' * U * S2) \ S2' * U * S2; %E in the paper
    D = A1 + C2' * A2 * C2 + (S1 - S2 * C2)' * U * (S1 - S2 * C2) + lambda * L1 + lambda * C2' * L2 * C2; %C in the paper
    % clear L1 L2;
    %% ��⼣��ʽ�Ĳ���
    D = (D + D') / 2;
    [v, eigval] = eig(D);
    eigval = diag(eigval);

    [eigval, idx] = sort(eigval); %�������У�=sort(eigval,1)
    Y1 = v(:, idx(1:maxbits)); %F in the paper
    Y2 = C2 * Y1; %���Y2
    %% Get W1 and W2
    eyemat_m1 = eye(m1);
    eyemat_m2 = eye(m2);
    W1 = (X1 * X1' + beta * eyemat_m1) \ X1 * Y1;
    W2 = (X2 * X2' + beta * eyemat_m2) \ X2 * Y2;

    %% Start to query
    [feaDimX, vid_num] = size(feaTrain_vis');
    %     oneline = ones(vid_num,1);
    X1_lowD = feaTrain_vis * W1; %low dimensity kf
    X1_med = median(X1_lowD);
    X1_binary = (X1_lowD > repmat(X1_med, vid_num, 1));
    B1 = X1_binary;
    %�����������Լ��ӵ�
    [feaDimX, text_num] = size(feaTrain_text');
    X2_lowD = feaTrain_text * W2; %low dimensity kf
    X2_med = median(X2_lowD);
    X2_binary = (X2_lowD > repmat(X2_med, text_num, 1));
    B2 = X2_binary;
    B.B1 = B1;
    B.B2 = B2;
    model.W1 = W1;
    model.W2 = W2;
    model.X1_med = X1_med;
    model.X2_med = X2_med;
    model.Y1 = Y1;
    model.Y2 = Y2;
end
