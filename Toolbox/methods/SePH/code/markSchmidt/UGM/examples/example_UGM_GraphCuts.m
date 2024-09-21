%% Make noisy X

getNoisyX;

%% Independent Decoding

[junk IndDecoding] = max(nodePot,[],2);

figure
imagesc(sign(X-.5));
colormap gray
title('Independent Decoding of Noisy X');

%% Decoding with ICM

fprintf('Running ICM decoding...\n');
ICMDecoding = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);

figure
imagesc(reshape(ICMDecoding,nRows,nCols));
colormap gray
title('ICM Decoding of Noisy X');
fprintf('(paused)\n');
pause

%% Decoding with Graph Cuts

fprintf('Running Graph Cut decoding...\n');
optimalDecoding = UGM_Decode_GraphCut(nodePot,edgePot,edgeStruct);

figure
imagesc(reshape(optimalDecoding,nRows,nCols));
colormap gray
title('Optimal Decoding of Noisy X');