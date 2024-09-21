function [XWtrueTestTraining,YWtrueTestTraining,GroundTruth]=Define_GrounDruth(XTrain,YTrain,XTest,YTest,LTrain,LTest)
    [nX,Xdim]=size(XTrain);
    [nY,Ydim]=size(YTrain);
    XaverageNumberNeighbors = 300;    % Xground truth is 50 nearest neighbor
    YaverageNumberNeighbors = 500;    % Yground truth is 50 nearest neighbor
    GroundTruth = (LTest * LTrain' >= 1);
    
     % Image_VS_Text
     % define ground-truth neighbors (this is only used for the evaluation):
    XR = randperm(nX);
    XDtrueTraining = distMat(XTrain(XR(1:200),:),XTrain); % sample 100 points to find a threshold
    XDball = sort(XDtrueTraining,2);
    clear XDtrueTraining;
    XDball = mean(XDball(:,XaverageNumberNeighbors));
    % scale data so that the target distance is 1
    XTrain = XTrain / XDball;
    XTest = XTest / XDball;
    XDball = 1;
    % threshold to define ground truth
    XDtrueTestTraining = distMat(XTest,XTrain);
    XWtrueTestTraining = XDtrueTestTraining < XDball;
    clear XDtrueTestTraining

     % Text_VS_Image
    % define ground-truth neighbors (this is only used for the evaluation):
    YR = randperm(nY);
    YDtrueTraining = distMat(YTrain(YR(1:300),:),YTrain); % sample 100 points to find a threshold
    YDball = sort(YDtrueTraining,2);
    clear YDtrueTraining;
    YDball = mean(YDball(:,YaverageNumberNeighbors));
    % scale data so that the target distance is 1
    YTrain = YTrain / YDball;
    YTest = YTest / YDball;
    YDball = 1;
    % threshold to define ground truth
    YDtrueTestTraining = distMat(YTest,YTrain);
    YWtrueTestTraining = YDtrueTestTraining < YDball;
    clear YDtrueTestTraining
end