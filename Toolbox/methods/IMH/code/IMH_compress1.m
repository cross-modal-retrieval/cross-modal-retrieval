function B = IMH_compress1(feaTest_text, model)

    % Start to query
    W2 = model.W2;
    X2_med = model.X2_med;
    [feaDimX, vid_num] = size(feaTest_text');
    %     oneline = ones(vid_num,1);
    X2_lowD = feaTest_text * W2; %low dimensity kf
    X2_med = median(X2_lowD);
    X2_binary = (X2_lowD > repmat(X2_med, vid_num, 1));

    B = X2_binary;

end
