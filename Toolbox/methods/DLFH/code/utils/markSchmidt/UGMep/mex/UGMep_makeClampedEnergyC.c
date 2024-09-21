#include <math.h>
#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Variables */
    int n,n1,n2,s,s1,s2,nodeNum,e,Vind,edgeNum,
            nNodes,nFree,nEdges,maxState,nInducedEdges,
            *nStates,*edgeEnds,*V,*E,*clamped;
    double *nodeEnergy, *edgeEnergy,*edgeWeights,*newNodeEnergy,*nodeMap,*newNstates,*newEdgeWeights,*newEdgeEnds,*edgeMap;
	
	/* Inputs */
    nodeEnergy = mxGetPr(prhs[0]);
    edgeEnergy = mxGetPr(prhs[1]);
	edgeWeights = mxGetPr(prhs[2]);
    nStates = (int*)mxGetPr(prhs[3]);
    edgeEnds = (int*)mxGetPr(prhs[4]);
    V = (int*)mxGetPr(prhs[5]);
    E = (int*)mxGetPr(prhs[6]);
    clamped = (int*)mxGetPr(prhs[7]);
    
    /* Sizes */
    nNodes = mxGetDimensions(prhs[0])[0];
    maxState = mxGetDimensions(prhs[0])[1];
    nEdges = mxGetDimensions(prhs[4])[0];
    
    /* Compute size of induced sub-graph */
    nFree = 0;
    for(n=0;n < nNodes;n++) {
        if(clamped[n]==-1)
            nFree++;
    }
    nInducedEdges = 0;
    for(e=0;e < nEdges; e++) {
        n1 = edgeEnds[e];
        n2 = edgeEnds[e+nEdges];
        if(clamped[n1]==-1 && clamped[n2]==-1)
            nInducedEdges++;
    }
	
    /* Output */
    plhs[0] = mxCreateDoubleMatrix(nFree,maxState,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nNodes,1,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(nFree,1,mxREAL);
    plhs[3] = mxCreateDoubleMatrix(nInducedEdges,1,mxREAL);
	plhs[4] = mxCreateDoubleMatrix(nInducedEdges,2,mxREAL);
	plhs[5] = mxCreateDoubleMatrix(nEdges,1,mxREAL);
    newNodeEnergy = mxGetPr(plhs[0]);
    nodeMap = mxGetPr(plhs[1]);
    newNstates = mxGetPr(plhs[2]);
    newEdgeWeights = mxGetPr(plhs[3]);
	newEdgeEnds = mxGetPr(plhs[4]);
    edgeMap = mxGetPr(plhs[5]);
	
    nodeNum = 0;
    for(n=0;n < nNodes;n++) {
        if(clamped[n]==-1) {
            
            /* Grab node potential */
            for(s=0;s < nStates[n];s++) {
                newNodeEnergy[nodeNum + nFree*s] = nodeEnergy[n + nNodes*s];
            }
            
            /* Absorb edge potentials from clamped neighbors */
            for(Vind = V[n]; Vind < V[n+1]; Vind++) {
                e = E[Vind];
                n1 = edgeEnds[e];
                n2 = edgeEnds[e+nEdges];
                
                if(n==n1) {
                    if(clamped[n2]!=-1) {
                        for(s=0;s<nStates[n];s++) {
                            newNodeEnergy[nodeNum + nFree*s] += edgeWeights[e]*edgeEnergy[s+maxState*clamped[n2]];
                        }
                    }
                }
                else {
                    if(clamped[n1]!=-1) {
                        for(s=0;s<nStates[n];s++) {
                            newNodeEnergy[nodeNum + nFree*s] += edgeWeights[e]*edgeEnergy[clamped[n1]+maxState*s];
                        }
                    }
                }
            }
            nodeMap[n] = nodeNum+1;
            newNstates[nodeNum] = nStates[n];
            nodeNum++;
        }
    }
    
    /* Grab edges in subgraph */
    edgeNum = 0;
    for(e=0;e < nEdges; e++) {
        n1 = edgeEnds[e];
        n2 = edgeEnds[e+nEdges];
        if(clamped[n1]==-1 && clamped[n2]==-1) {
			for(s1=0;s1<nStates[n1];s1++) {
                for(s2=0;s2<nStates[n2];s2++) {
                    newEdgeWeights[edgeNum] = edgeWeights[e];
                }
            }
			newEdgeEnds[edgeNum] = nodeMap[n1];
			newEdgeEnds[edgeNum+nInducedEdges] = nodeMap[n2];
			edgeMap[e] = edgeNum+1;
            edgeNum++;
        }
    }
}
