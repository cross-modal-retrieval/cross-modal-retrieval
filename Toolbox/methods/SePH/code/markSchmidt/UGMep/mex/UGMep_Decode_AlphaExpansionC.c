#include <math.h>
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Variables */
	int n,n1,n2,e,nNodes,maxState,nEdges,*edgeEnds,s,*y,*swapPositions,sizeNewEdgeEnergy[3];
	double *nodeEnergy,*edgeEnergy,*edgeWeights,*newNodeEnergy,*newEdgeEnergy;
	
	/* Inputs */
	nodeEnergy = mxGetPr(prhs[0]);
	edgeEnergy = mxGetPr(prhs[1]);
	edgeWeights = mxGetPr(prhs[2]);
	edgeEnds = (int*)mxGetPr(prhs[3]);
	s = (int)mxGetScalar(prhs[4]);
	y = (int*)mxGetPr(prhs[5]);
	swapPositions = (int*)mxGetPr(prhs[6]);
	
	nNodes = mxGetDimensions(prhs[0])[0];
	maxState = mxGetDimensions(prhs[0])[1];
	nEdges = mxGetDimensions(prhs[3])[0];

	/* Outpus */
	sizeNewEdgeEnergy[0] = 2;
	sizeNewEdgeEnergy[1] = 2;
	sizeNewEdgeEnergy[2] = nEdges;
	plhs[0] = mxCreateDoubleMatrix(nNodes,2,mxREAL);
	plhs[1] = mxCreateNumericArray(3,sizeNewEdgeEnergy,mxDOUBLE_CLASS,mxREAL);
	newNodeEnergy = mxGetPr(plhs[0]);
	newEdgeEnergy = mxGetPr(plhs[1]);
	
	for(n=0;n<nNodes;n++) {
		newNodeEnergy[n] = nodeEnergy[n+nNodes*s];
		newNodeEnergy[n+nNodes] = nodeEnergy[n+nNodes*y[swapPositions[n]]];
	}
	
	for(e=0;e<nEdges;e++) {
		n1 = edgeEnds[e];
		n2 = edgeEnds[e+nEdges];
		newEdgeEnergy[0+2*(0+2*e)] = edgeWeights[e]*edgeEnergy[s+maxState*s];
		newEdgeEnergy[0+2*(1+2*e)] = edgeWeights[e]*edgeEnergy[s+maxState*y[swapPositions[n2]]];
		newEdgeEnergy[1+2*(0+2*e)] = edgeWeights[e]*edgeEnergy[y[swapPositions[n1]]+maxState*s];
		newEdgeEnergy[1+2*(1+2*e)] = edgeWeights[e]*edgeEnergy[y[swapPositions[n1]]+maxState*y[swapPositions[n2]]];
	}
}
