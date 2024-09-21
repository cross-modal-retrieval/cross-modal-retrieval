function [XKTrain, XKTest, XKRetrieval] = Kernel_Feature_eval(XTrain, XTest, XRetrieval, Anchors)
    [nX, Xdim] = size(XTrain);

    [nXT, XTdim] = size(XTest);

    [nXR, XRdim] = size(XRetrieval);

    XKTrain = sqdist(XTrain', Anchors');
    Xsigma = mean(mean(XKTrain, 2));
    XKTrain = exp(-XKTrain / (2 * Xsigma));
    Xmvec = mean(XKTrain);
    XKTrain = XKTrain - repmat(Xmvec, nX, 1);

    XKTest = sqdist(XTest', Anchors');
    XKTest = exp(-XKTest / (2 * Xsigma));
    XKTest = XKTest - repmat(Xmvec, nXT, 1);

    XKRetrieval = sqdist(XRetrieval', Anchors');
    XKRetrieval = exp(-XKRetrieval / (2 * Xsigma));
    XKRetrieval = XKRetrieval - repmat(Xmvec, nXR, 1);
end
