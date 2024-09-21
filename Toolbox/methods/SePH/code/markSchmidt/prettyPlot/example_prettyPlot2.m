% Demo shows how to:
%    1. use cell arrays when lines have different length
%    2. use repeating sequence of markers/colors/lineStyles
%    3. truncate axes to partially-specified limits

clear all
close all

%% Generate some data
maxIter = 200;
optTol = 1e-6;
err0 = 10000;
rho = .04;
rho2 = .04:.005:1;
for k = 1:maxIter
	err1(k) = err0/sqrt(k);
	if err1(k) < optTol
		break;
	end
end
for k = 1:maxIter
	err2(k) = err0/k;
	if err2(k) < optTol
		break;
	end
end
for k = 1:maxIter
	err3(k) = err0/k^2;
	if err3(k) < optTol
		break;
	end
end
for k = 1:maxIter
	err4(k) = err0*(1-rho)^k;
	if err4(k) < optTol
		break;
	end
end
for k = 1:maxIter
	err5(k) = err0*(1-sqrt(rho))^k;
	if err5(k) < optTol
		break;
	end
end
for k = 1:maxIter
	err6(k) = err0*prod(1-rho2(1:k));
	if err6(k) < optTol
		break;
	end
end

% Note: yData does not have same length, and is stored as cell array
xData = 1:maxIter;
yData{1} = err1;
yData{2} = err2;
yData{3} = err3;
yData{4} = err4;
yData{5} = err5;
yData{6} = err6;
legendStr = {'Sublinear1','Sublinear2','Sublinear3','Linear1','Linear2','Superlinear'};


%% Pretty Plot

figure;
options.logScale = 2;

% Note: prettyPlot will cycle through the given colors/lineStyles/markers
options.colors = [1 0 0
	0 1 0
	0 0 1];
options.lineStyles = {':','--','-'};
options.markers = {'o','s'};


% Note: we will truncate the lower y-axis limit but not the upper limit
options.ylimits = [1e-6 inf];

options.markerSpacing = [25 1
	25 11
	25 21
	25 5
	25 15
	25 8];
options.xlabel = 'Iteration Number';
options.ylabel = 'Error';
options.legend = legendStr;
options.legendLoc = 'SouthWest';
prettyPlot(xData,yData,options);
print -dpdf prettyPlot2.pdf