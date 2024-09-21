
clear all

% Load data
fprintf('Loading OCR data...\n');
X = load('OCR_features.data');
Y = load('OCR_labels.data');

% Put into UGM format
fprintf('Putting OCR data into UGM format\n');
nFeatures = size(X,2);
nStates = max(Y);
globalFeatures = [1;zeros(nFeatures,1)];
ising = 0;
tied = 1;
paramLastState = 1;

exampleEnds = [find(Y==0)-1];
exampleStarts = [1;find(Y==0)+1];
nExamples = length(exampleEnds);
examples = cell(nExamples,1);
for i = 1:nExamples
    exampleInd = exampleStarts(i):exampleEnds(i);
    nNodes = length(exampleInd);
    examples{i}.edgeStruct = UGM_makeEdgeStruct(chainAdjMatrix(nNodes),nStates);
    examples{i}.Y = int32(Y(exampleInd))';
    for n = 1:nNodes
        examples{i}.Xnode(1,:,n) = [1 X(exampleInd(n),:)];
    end
    examples{i}.Xedge = ones(1,1,nNodes-1);
    [examples{i}.nodeMap examples{i}.edgeMap w] = UGM_makeCRFmaps(examples{i}.Xnode,examples{i}.Xedge,examples{i}.edgeStruct,ising,tied,paramLastState);
end

% Split into training/testing data
fprintf('Splitting into training and testing data\n');
fold = load('OCR_fold.data');
trainNdx = fold(exampleStarts(1:end-1))==0;
testNdx = fold(exampleStarts(1:end-1))~=0;
trainExamples = examples(trainNdx);
testExamples = examples(testNdx);

% Train L2-regularized CRF
fprintf('Finding maximum likelihood parameters\n');
UGM_CRFcell_NLL(w,trainExamples,@UGM_Infer_Chain);
funObj = @(w)UGM_CRFcell_NLL(w,trainExamples,@UGM_Infer_Chain);
lambda = ones(size(w));
penalizedFunObj = @(w)penalizedL2(w,funObj,lambda);
options.Display = 'full';
w = minFunc(penalizedFunObj,w,options);

% Compute test error
testErrs = 0;
Z = 0;
for i = 1:length(testExamples)
    [nodePot,edgePot] = UGM_CRF_makePotentials(w,testExamples{i}.Xnode,testExamples{i}.Xedge,testExamples{i}.nodeMap,testExamples{i}.edgeMap,testExamples{i}.edgeStruct);
    yMAP = UGM_Decode_Chain(nodePot,edgePot,testExamples{i}.edgeStruct);
    
    testErrs = testErrs + sum(yMAP'~=testExamples{i}.Y);
    Z = Z + length(testExamples{i}.Y);
end
testErrorRate = testErrs/Z