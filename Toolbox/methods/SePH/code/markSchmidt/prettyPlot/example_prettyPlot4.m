clear all
load SAGexperiment.mat

options.legendLoc = 'NorthEast';
options.logScale = 2;
options.colors = colors;
options.lineStyles = lineStyles;
options.markers = markers;
options.markerSize = 12;
options.markerSpacing = markerSpacing;
options.legendStr = names;
options.legend = names;
options.ylabel = 'Objective minus Optimum';
options.xlabel = 'Effective Passes';
options.labelLines = 1;
options.labelRotate = 1;
options.xlimits = [0 maxIter];
options.ylimits = [-inf 1.01];

figure;
prettyPlot(fEvals,fVals,options);
print -dpdf prettyPlot4.pdf