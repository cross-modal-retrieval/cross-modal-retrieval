#include <math.h>
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Variables */
	int n1,n2,e,nNodes,nEdges,*edgeEnds;
	double *nodeEnergy,*edgeEnergy,*edgeWeights;
	
	/* Inputs */
	nodeEnergy = mxGetPr(prhs[0]);
	edgeEnergy = mxGetPr(prhs[1]);
	edgeWeights = mxGetPr(prhs[2]);
	edgeEnds = (int*)mxGetPr(prhs[3]);
	
	nNodes = mxGetDimensions(prhs[0])[0];
	nEdges = mxGetDimensions(prhs[3])[0];
	
	for(e=0;e<nEdges;e++) {
		n1 = edgeEnds[e];
		n2 = edgeEnds[e+nEdges];
		nodeEnergy[n1+nNodes] += edgeWeights[e]*(edgeEnergy[1 + 2*0] - edgeEnergy[0 + 2*0]);
		nodeEnergy[n2+nNodes] += edgeWeights[e]*(edgeEnergy[1 + 2*1] - edgeEnergy[1 + 2*0]); 
	}
}
