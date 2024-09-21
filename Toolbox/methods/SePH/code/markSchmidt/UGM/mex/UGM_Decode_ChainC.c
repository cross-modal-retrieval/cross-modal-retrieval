#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   /* Variables */
    int n, s,s1,s2,e, nEdges,
    nNodes, maxState,*nStates,*maximizers,*y,dims[2];
    
    double kappa,*tmp,max_tmp,
    *nodePot, *edgePot, *alpha;
    
   /* Input */
    
    nodePot = mxGetPr(prhs[0]);
    edgePot = mxGetPr(prhs[1]);
    nStates = (int*)mxGetPr(prhs[2]);
	
	if (!mxIsClass(prhs[2],"int32"))
        mexErrMsgTxt("nStates must be int32");
    
   /* Compute Sizes */
    
    nNodes = mxGetDimensions(prhs[0])[0];
    maxState = mxGetDimensions(prhs[0])[1];
    nEdges = nNodes-1;
    
   /* Output */
	dims[0] = nNodes;
	dims[1] = 1;
	plhs[0] = mxCreateNumericArray(2,dims,mxINT32_CLASS,mxREAL);
    y = mxGetData(plhs[0]);
    
   /* Initialize */
    alpha = mxCalloc(nNodes*maxState,sizeof(double));
    tmp = mxCalloc(maxState*maxState,sizeof(double));
	maximizers = mxCalloc(nNodes*maxState,sizeof(double));
    
	kappa = 0;
    for(s = 0; s < nStates[0]; s++) {
        alpha[nNodes*s] = nodePot[nNodes*s];
        kappa += alpha[nNodes*s];
    }
    for(s = 0; s < nStates[0]; s++) {
        alpha[nNodes*s] /= kappa;
    }
    
   /* Forward Pass */
	kappa = 0;
    for(n = 1; n < nNodes;n++) {
        for(s1 = 0; s1 < nStates[n-1];s1++) {
            for(s2 = 0; s2 < nStates[n];s2++) {
                tmp[s1 + maxState*s2] = alpha[n-1 + nNodes*s1]*edgePot[s1 + maxState*(s2 + maxState*(n-1))];
            }
        }
        for(s2 = 0; s2 < nStates[n];s2++) {
            max_tmp = 0;
            for(s1 = 0; s1 < nStates[n-1];s1++) {
				if(tmp[s1 + maxState*s2] >= max_tmp) {
					maximizers[n + nNodes*s2] = s1;
					max_tmp = tmp[s1 + maxState*s2];
				}
            }
            alpha[n + nNodes*s2] = nodePot[n + nNodes*s2]*max_tmp;
            kappa += alpha[n + nNodes*s2];
        }
        for(s = 0; s < nStates[n];s++) {
            alpha[n + nNodes*s] /= kappa;
        }
    }
    
    /* Backward Pass */
	max_tmp = 0;
    for(s = 0; s < nStates[nNodes-1]; s++)
    {
		if(alpha[nNodes-1 + nNodes*s] > max_tmp) {
			max_tmp = alpha[nNodes-1 + nNodes*s];
			y[nNodes-1] = s;
		}
	}
	
    for(n = nNodes-2; n >= 0; n--)
		y[n] = maximizers[n+1 + nNodes*(y[n+1])];
	
	for(n=0;n<nNodes;n++)
		y[n]++;
    
   /* Free memory */
    mxFree(alpha);
    mxFree(tmp);
	mxFree(maximizers);
}
