#include <math.h>
#include "mex.h"
#include "UGM_common.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int n,e,s,s1,s2,f,fNum,*nodeMap,*edgeMap,nStates,nNodes,nEdges,nNodeFeatures,nEdgeFeatures,ising,tied,paramLastState;
    
    if (nrhs != 5)
        mexErrMsgTxt("Expected 5 arguments:\nUGM_makeCRFMapsC(nodeMap,edgeMap,ising,tied,paramLastState)");
    
    if (!mxIsClass(prhs[0],"int32")||!mxIsClass(prhs[1],"int32")||!mxIsClass(prhs[2],"int32")||
            !mxIsClass(prhs[3],"int32")||!mxIsClass(prhs[4],"int32"))
        mexErrMsgTxt("all input arguments must be int32");
    
    /* Input */
    nodeMap = (int*)mxGetPr(prhs[0]);
    edgeMap = (int*)mxGetPr(prhs[1]);
    ising = (int)mxGetScalar(prhs[2]);
    tied = (int)mxGetScalar(prhs[3]);
    paramLastState = (int)mxGetScalar(prhs[4]);
    
    /* Compute Sizes */
    nNodes = mxGetDimensions(prhs[0])[0];
    nStates = mxGetDimensions(prhs[0])[1];
    if (mxGetNumberOfDimensions(prhs[0]) == 3)
        nNodeFeatures = mxGetDimensions(prhs[0])[2];
    else
        nNodeFeatures = 1;
    nEdges = mxGetDimensions(prhs[1])[2];
    if (mxGetNumberOfDimensions(prhs[1]) == 4)
        nEdgeFeatures = mxGetDimensions(prhs[1])[3];
    else
        nEdgeFeatures = 1;
    
    /* Make nodeMap */
    fNum = 1;
    for(f=0;f<nNodeFeatures;f++) {
        for(s = 0;s < nStates; s++) {
            if (!paramLastState && s == nStates-1)
                break;
            
            for(n = 0;n < nNodes; n++) {
                nodeMap[n + nNodes*(s + nStates*f)] = fNum;
                if (!tied)
                    fNum++;
            }
            if (tied)
                fNum++;
        }
    }
    
    /* Make edgeMap */
    for(f=0;f<nEdgeFeatures;f++) {
        if (tied) {
            for(s1 = 0;s1 < nStates; s1++) {
                if (ising == 1) {
                    for(e = 0;e < nEdges; e++) {
                        edgeMap[s1 + nStates*(s1 + nStates*(e + nEdges*f))] = fNum;
                    }
                }
                else if (ising == 2) {
                    for(e = 0;e < nEdges; e++) {
                        edgeMap[s1 + nStates*(s1 + nStates*(e + nEdges*f))] = fNum;
                    }
                    fNum++;
                }
                else {
                    for(s2 = 0;s2 < nStates;s2++) {
                        for(e = 0;e < nEdges; e++) {
                            edgeMap[s1 + nStates*(s2 + nStates*(e + nEdges*f))] = fNum;
                        }
                        fNum++;
                    }
                }
            }
            if (ising == 1)
                fNum++;
        }
        else {
            for(e = 0;e < nEdges;e++) {
                for(s1 = 0;s1 < nStates;s1++) {
                    if (ising == 1) {
                        edgeMap[s1 + nStates*(s1 + nStates*(e + nEdges*f))] = fNum;
                    }
                    else if (ising == 2) {
                        edgeMap[s1 + nStates*(s1 + nStates*(e + nEdges*f))] = fNum++;
                    }
                    else {
                        for(s2 = 0;s2 < nStates;s2++) {
                            edgeMap[s1 + nStates*(s2 + nStates*(e + nEdges*f))] = fNum++;
                        }
                    }
                }
                if (ising == 1)
                    fNum++;
            }
        }
    }
     
}