function [samples] = UGM_Sample_Conditional(nodePot,edgePot,edgeStruct,clamped,sampleFunc,varargin)
% Do sampling with observed values

[nNodes,maxState] = size(nodePot);
nEdges = size(edgePot,3);
edgeEnds = edgeStruct.edgeEnds;
nSamples = edgeStruct.maxIter;

[clampedNP,clampedEP,clampedES,edgeMap] = UGM_makeClampedPotentials(nodePot,edgePot,edgeStruct,clamped);

clampedSamples = sampleFunc(clampedNP,clampedEP,clampedES,varargin{:});

samples = repmat(clamped(:)',[nSamples 1]);
samples(samples==0) = clampedSamples;