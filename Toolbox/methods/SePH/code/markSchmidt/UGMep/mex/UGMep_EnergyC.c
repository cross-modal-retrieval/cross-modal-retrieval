#include <math.h>
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    
    int n, n1, n2, e,
            nNodes, nEdges, maxState,
            *edgeEnds, *y;
    double *nodeEnergy, *edgeEnergy, *edgeWeights,*energy;
    
    /* Input */
    y = (int*)mxGetPr(prhs[0]);
    nodeEnergy = mxGetPr(prhs[1]);
    edgeEnergy = mxGetPr(prhs[2]);
	edgeWeights = mxGetPr(prhs[3]);
    edgeEnds = (int*)mxGetPr(prhs[4]);
    
    /* Compute Sizes */
    nNodes = mxGetDimensions(prhs[1])[0];
    maxState = mxGetDimensions(prhs[1])[1];
    nEdges = mxGetDimensions(prhs[4])[0];
    
    /* Output */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    energy = mxGetPr(plhs[0]);
    
    *energy = 0;
    for(n = 0; n < nNodes; n++) {
        *energy += nodeEnergy[n + nNodes*y[n]];
    }
    for(e = 0; e < nEdges; e++) {
        n1 = edgeEnds[e];
        n2 = edgeEnds[e+nEdges];
        *energy += edgeWeights[e]*edgeEnergy[y[n1] + maxState*y[n2]];
    }
    
    
}
