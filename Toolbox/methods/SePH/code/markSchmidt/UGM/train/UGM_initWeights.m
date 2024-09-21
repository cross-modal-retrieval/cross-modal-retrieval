function w = UGM_initWeights(nodeMap,edgeMap)

nParams = max(max(nodeMap(:)),max(edgeMap(:)));
w = zeros(nParams,1);