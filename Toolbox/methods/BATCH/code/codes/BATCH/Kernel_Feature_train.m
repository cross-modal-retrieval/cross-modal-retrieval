function [XKTrain] = Kernel_Feature_train(XTrain, Anchors)

    [nX, Xdim] = size(XTrain);

    XKTrain = sqdist(XTrain', Anchors');
    Xsigma = mean(mean(XKTrain, 2));
    XKTrain = exp(-XKTrain / (2 * Xsigma));
    Xmvec = mean(XKTrain);
    XKTrain = XKTrain - repmat(Xmvec, nX, 1);

end
