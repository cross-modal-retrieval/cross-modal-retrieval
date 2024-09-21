function [XKTrain] = Kernel_Feature(XTrain, Anchors)

    [nX, Xdim] = size(XTrain);

    [nXT, XTdim] = size(XTest);

    [nXR, XRdim] = size(XRetrieval);

    XKTrain = sqdist(XTrain', Anchors');
    Xsigma = mean(mean(XKTrain, 2));
    XKTrain = exp(-XKTrain / (2 * Xsigma));
    Xmvec = mean(XKTrain);
    XKTrain = XKTrain - repmat(Xmvec, nX, 1);

end
