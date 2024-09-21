function evaluation_info=evaluate(XTrain,YTrain,XTest,YTest,LTest,LTrain,GroundTruth,M,param,retrieve_data)
     
    tmp_T=cputime;  %���¼���ʱ��
    
    [Wx Wy R B] = train(XTrain, YTrain, param, LTrain);
    
    traintime=cputime-tmp_T;  % Training Time
    evaluation_info.trainT=traintime;   %����train��ʱ��
    
    %      % X��Ϊ���� YΪ��ݿ�
    BxTest = compactbit(XTest*Wx'*R' >= 0);
%     ByTrain = compactbit(B' >= 0);
    ByTrain = compactbit(retrieve_data.T_base* Wy'*R' >= 0);
    DHamm = hammingDist(BxTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
    
    evaluation_info.Image_VS_Text_MAP = mAP(orderH', retrieve_data.L_base, LTest);
    [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', retrieve_data.L_base, LTest);
    evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', retrieve_data.L_base, LTest);

% YΪ���� XΪ��ݿ�
    ByTest = compactbit(YTest*Wy'*R' >= 0);
    BxTrain = compactbit(retrieve_data.I_base* Wx'*R' >= 0);
    DHamm = hammingDist(ByTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', retrieve_data.L_base, LTest);
    [evaluation_info.Text_VS_Image_precision, evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', retrieve_data.L_base, LTest);
    evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', retrieve_data.L_base, LTest);
    
    compressiontime=cputime-tmp_T;
    evaluation_info.compressT=compressiontime;     %����hashѹ����ʱ��
    
end