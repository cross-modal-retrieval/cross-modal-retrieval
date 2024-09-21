#include <math.h>
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Variables */
	int n, s, n1, n2,e,Vind,
			nNodes, nEdges, maxState,*nStates,done,
			*y,
			*edgeEnds,*V,*E,minY;
	
	double *energy,*nodeEnergy, *edgeEnergy, *edgeWeights, *yMAP,minEnergy;
	
	/* Input and Sizes */
	y = (int*)mxGetPr(prhs[0]);
	nodeEnergy = mxGetPr(prhs[1]);
	edgeEnergy = mxGetPr(prhs[2]);
	edgeWeights = mxGetPr(prhs[3]);
	edgeEnds = (int*)mxGetPr(prhs[4]);
    V = (int*)mxGetPr(prhs[5]);
    E = (int*)mxGetPr(prhs[6]);
    nStates = (int*)mxGetPr(prhs[7]);
    
    /* Compute Sizes */
	nNodes = mxGetDimensions(prhs[1])[0];
    maxState = mxGetDimensions(prhs[1])[1];
    nEdges = mxGetDimensions(prhs[4])[0];
	    
    /* Allocate memory */
    energy = mxCalloc(maxState,sizeof(double));
    
    /* Output */
    plhs[0] = mxCreateDoubleMatrix(nNodes,1,mxREAL);
    yMAP = mxGetPr(plhs[0]);
    
    done = 0;
    while(!done) {
        done = 1;
        
        for(n=0; n<nNodes; n++) {
            for(s=0; s<nStates[n]; s++)
                energy[s] = nodeEnergy[n+nNodes*s];
            
            for(Vind = V[n]; Vind < V[n+1]; Vind++)
            {
                e = E[Vind];
                n1 = edgeEnds[e];
                n2 = edgeEnds[e+nEdges];
                
                if(n==n1) {
                    for(s=0; s<nStates[n]; s++)
                        energy[s] += edgeWeights[e]*edgeEnergy[s+maxState*y[n2]];
                }
                else {
                    for(s=0; s<nStates[n]; s++)
                        energy[s] += edgeWeights[e]*edgeEnergy[y[n1]+maxState*s];
                }
            }
            
            minEnergy = energy[0];
            minY = 0;
            for(s=1;s<nStates[n];s++)
            {
                if(energy[s] < minEnergy) {
                    minEnergy = energy[s];
                    minY = s;
                }
            }
            if(minY != y[n]) {
                y[n] = minY;
                done = 0;
            }
        }
    }
    
    for(n=0;n<nNodes;n++)
    {
        yMAP[n] = y[n]+1;
    }
    mxFree(energy);
}
