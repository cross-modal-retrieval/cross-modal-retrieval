%% Demo shows how to plot upper/lower error lines

clear all
close all

%% Load data
load timeProjection1_0.10.mat

figure;
xData = elementRange;
for i = 1:3
	RT = runTime(:,:,i);
	
	yData{i} = mean(RT,2);
	errors{i,1} = min(RT,[],2);
	errors{i,2} = max(RT,[],2);
end

options.errors = errors;
options.errorStyle = {'--'};
options.errorColors = [.75 .75 1
	.75 1 .75
	1 .75 .75];

options.colors = [0 0 .5
	0 .5 0
	.5 0 0];
options.title = sprintf('Projection Time against Vector Length (Sparsity = %d%%))',kRatio*100);
options.xlabel = 'Vector Length';
options.ylabel = 'Runtime (s)';
options.legend = {'Random','Heap','Sort'};
options.legendLoc = 'NorthWest';
options.labelLines = 1;
prettyPlot(xData,yData,options);
print -dpdf prettyPlot3.pdf