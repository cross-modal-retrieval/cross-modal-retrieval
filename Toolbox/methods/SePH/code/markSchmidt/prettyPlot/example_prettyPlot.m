clear all
close all

%% Generate some data
maxIter = 100;
err0 = 10000;
err1 = zeros(1,100);
err2 = zeros(1,100);
err3 = zeros(1,100);
err4 = zeros(1,100);
err5 = zeros(1,100);
err6 = zeros(1,100);
rho = .04;
rho2 = .04:.005:1;
for k = 1:maxIter
	err1(k) = err0/sqrt(k);
	err2(k) = err0/k;
	err3(k) = err0/k^2;
	err4(k) = err0*(1-rho)^k;
	err5(k) = err0*(1-sqrt(rho))^k;
	err6(k) = err0*prod(1-rho2(1:k));
end

%% Matlab way
xData = 1:maxIter;
yData = [err1;err2;err3;err4;err5;err6];
legendStr = {'Sublinear1','Sublinear2','Sublinear3','Linear1','Linear2','Superlinear'};

figure(1);
plot(xData,yData);
legend(legendStr);
title('Plot made with plot function (press any key to go to next step)');
pause

clf;
semilogy(xData,yData);
legend(legendStr);
title('Plot made with plot function (press any key to go to next step)');
pause

clf;
semilogy(xData,err1,'r:');
hold on;
semilogy(xData,err2,'r--');
semilogy(xData,err3,'r-');
semilogy(xData,err4,'g--');
semilogy(xData,err5,'g-');
semilogy(xData,err6,'b-');
legend(legendStr);
title('Plot made with plot function (press any key to go to next step)');
pause

clf;
semilogy(xData,err1,'ro:');
hold on;
semilogy(xData,err2,'ro--');
semilogy(xData,err3,'ro-');
semilogy(xData,err4,'gs--');
semilogy(xData,err5,'gs-');
semilogy(xData,err6,'bp-');
legend(legendStr,'Location','SouthWest');
xlabel('Iteration Number');
ylabel('Error');
title('Plot made with plot function');
print -dpdf regularPlot.pdf;
fprintf('(press any key to start using prettyPlot)\n');
pause

%% Pretty Plot

figure;
prettyPlot(xData,yData);
title('Plot made with prettyPlot function (press any key to go to next step)');
pause

clf;
options.logScale = 2;
options.title = 'Plot made with prettyPlot function (press any key to go to next step)';
prettyPlot(xData,yData,options);
pause

clf;
options.colors = [1 0 0
	1 0 0
	1 0 0
	0 1 0
	0 1 0
	0 0 1];
options.lineStyles = {':','--','-','--','-','-'};
options.legend = legendStr;
prettyPlot(xData,yData,options);
pause

clf;
options.markers = {'o','o','o','s','s','p'};
options.xlabel = 'Iteration Number';
options.ylabel = 'Error';
options.legendLoc = 'SouthWest';
prettyPlot(xData,yData,options);
pause

clf;
options.markerSpacing = [25 1
	25 1
	25 1
	25 1
	25 1
	25 1];
options.xlabel = 'Iteration Number';
options.ylabel = 'Error';
options.legendLoc = 'SouthWest';
prettyPlot(xData,yData,options);
pause

clf;
options.markerSpacing = [25 1
	25 11
	25 21
	25 5
	25 15
	25 8];
options.title = 'Final plot made using prettyPlot';
prettyPlot(xData,yData,options);
print -dpdf prettyPlot.pdf