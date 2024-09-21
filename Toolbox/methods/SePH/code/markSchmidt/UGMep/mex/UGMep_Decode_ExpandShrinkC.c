#include <math.h>
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Variables */
	int n, s, s1, s2, n1, n2, e,y1,y2,
			nNodes, nEdges, maxState,
			*y,
			*edgeEnds;
	
	double *nodeEnergy, *edgeEnergy, *modifiedNE, *modifiedEE, *edgeWeights,tmp;
	
	/* Input */
	
	y = (int*)mxGetPr(prhs[0]);
	s1 = (int)mxGetScalar(prhs[1]);
	s2 = (int)mxGetScalar(prhs[2]);
	nodeEnergy = mxGetPr(prhs[3]);
	edgeEnergy = mxGetPr(prhs[4]);
	edgeWeights = mxGetPr(prhs[5]);
	edgeEnds = (int*)mxGetPr(prhs[6]);
	modifiedNE = mxGetPr(prhs[7]);
	modifiedEE = mxGetPr(prhs[8]);
	
	/* Compute Sizes */
	
	nNodes = mxGetDimensions(prhs[3])[0];
	maxState = mxGetDimensions(prhs[3])[1];
	nEdges = mxGetDimensions(prhs[6])[0];
	
	/* Make conditional node energies */
	for(n=0; n < nNodes; n++) {
		if (y[n]==s1) {
			modifiedNE[n] = nodeEnergy[n+nNodes*s1];
			modifiedNE[n+nNodes] = nodeEnergy[n+nNodes*s2];
		}
		else {
			modifiedNE[n] = nodeEnergy[n+nNodes*s1];
			modifiedNE[n+nNodes] = nodeEnergy[n+nNodes*y[n]];
		}
	}
	for(e=0; e < nEdges; e++) {
		n1 = edgeEnds[e];
		n2 = edgeEnds[e+nEdges];
		y1 = y[n1];
		y2 = y[n2];
		if(y1==s1)
			y1 = s2;
		if(y2==s1)
			y2 = s2;
		
		/* Make conditional edge energies */
		modifiedEE[4*e] = edgeWeights[e]*edgeEnergy[s1+s1*maxState]; /* (1,1) */
		modifiedEE[1+4*e] = edgeWeights[e]*edgeEnergy[y1+s1*maxState]; /* (2,1) */
		modifiedEE[2+4*e] = edgeWeights[e]*edgeEnergy[s1+y2*maxState]; /* (1,2) */
		modifiedEE[3+4*e] = edgeWeights[e]*edgeEnergy[y1+y2*maxState]; /* (2,2) */
		
		/* Modify to be sub-modular */
		if (y[n1] == s1 && y[n2] == s2) {
			tmp = modifiedEE[2+4*e]+modifiedEE[1+4*e]-modifiedEE[3+4*e];
			if (modifiedEE[4*e] > tmp)
				modifiedEE[4*e] = tmp;
		}
		else if (y[n1] == s1) {
			tmp = modifiedEE[4*e]+modifiedEE[3+4*e]-modifiedEE[2+4*e];
			if (modifiedEE[1+4*e] < tmp)
				modifiedEE[1+4*e] = tmp;
		}
		else if (y[n2] == s1) {
			tmp = modifiedEE[4*e]+modifiedEE[3+4*e]-modifiedEE[1+4*e];
			if (modifiedEE[2+4*e] < tmp)
				modifiedEE[2+4*e] = tmp;
		}
		else {
			tmp = modifiedEE[2+4*e]+modifiedEE[1+4*e]-modifiedEE[4*e];
			if (modifiedEE[3+4*e] > tmp)
				modifiedEE[3+4*e] = tmp;
		}
		
		/* Move energy from edges to nodes */
		modifiedNE[n1+nNodes] += modifiedEE[1+4*e] - modifiedEE[4*e];
		modifiedNE[n2+nNodes] += modifiedEE[3+4*e] - modifiedEE[1+4*e];
	}
}
