function B = IMH_compress(feaTest_vis, model)

    % Start to query
    W1 = model.W1;
    %X1_med=model.X1_med;
    [vis_num, feaDimX] = size(feaTest_vis);
    %     oneline = ones(vid_num,1);
    X1_lowD = feaTest_vis * W1; %low dimensity kf
    X1_med = median(X1_lowD);
    X1_binary = (X1_lowD > repmat(X1_med, vis_num, 1));

    B = X1_binary;

end
