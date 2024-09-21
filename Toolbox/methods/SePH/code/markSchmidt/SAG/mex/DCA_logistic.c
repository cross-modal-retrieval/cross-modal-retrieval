#include <math.h>
#include "mex.h"

/* 
 DCA_logistic(alpha,yXt,lambda,iVals[,w,yxxy]);
 */


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    int k;
    long i,j,nVars;
    double gi,Hi,alphaOld,c,d,UB,LB;
    mwIndex *jc, *ir;
    
    if (nrhs < 4)
        mexErrMsgTxt("At least 4 arguments are needed: {alpha,yXt,lambda,jVals[,w,yxxy]}");
    
    double *alpha = mxGetPr(prhs[0]);
    double *yXt = mxGetPr(prhs[1]);
    double lambda = mxGetScalar(prhs[2]);
    int *iVals = (int*)mxGetPr(prhs[3]);
    
    /* Compute Sizes */
    int nSamples = mxGetN(prhs[1]);
    nVars = mxGetM(prhs[1]);
    int maxIter = mxGetM(prhs[3]);
    
    /* Basic input checking */
    if (nSamples != mxGetM(prhs[0]) || 1 != mxGetN(prhs[0]))
        mexErrMsgTxt("alpha should be a column vector, with length equal to the number of columns of yXt");
    if (!mxIsClass(prhs[3],"int32") || 1 != mxGetN(prhs[3]))
        mexErrMsgTxt("iVals must be an int32 column vector");
    
    int sparse = 0;
    if (mxIsSparse(prhs[1])) {
        sparse = 1;
        jc = mxGetJc(prhs[1]);
        ir = mxGetIr(prhs[1]);
    }
        
    lambda = lambda*nSamples;
    
    
    /* Initialize w */
    double *w;
    if (nrhs >= 5) {
        w = mxGetPr(prhs[4]);
        if (nVars != mxGetM(prhs[4]) || 1 != mxGetN(prhs[4]))
            mexErrMsgTxt("w should be a column vector, with the same number of rows as yXt");
    }
    else {
        w = mxCalloc(nVars,sizeof(double));
        for(i = 0;i < nSamples;i++) {
            if (sparse) {
                for(j = jc[i];j < jc[i+1];j++) {
                    w[ir[j]] -= yXt[j]*alpha[i]/lambda;
                }
            }
            else
            {
                for(j = 0;j < nVars;j++) {
                    w[j] -= yXt[j + nVars*i]*alpha[i]/lambda;
                }
            }
        }
    }
            
    /* Compute yxxy */
    double *yxxy;
    if (nrhs >= 6) {
        yxxy = mxGetPr(prhs[5]);
        if (nSamples != mxGetM(prhs[5]) || 1 != mxGetN(prhs[5]))
            mexErrMsgTxt("yxxy should be a column vector, with length equal to the number of columns in yXt"); 
    }
    else {
        yxxy = mxCalloc(nSamples,sizeof(double));
        for(i = 0;i < nSamples;i++) {
            if (sparse) {
                for(j = jc[i];j < jc[i+1];j++) {
                    yxxy[i] += yXt[j]*yXt[j];
                }
            }
            else {
                for(j = 0;j < nVars;j++) {
                    yxxy[i] += yXt[j + nVars*i]*yXt[j + nVars*i];
                }
            }
        }
    }
    
    for(k=0;k<maxIter;k++)
    {
        /* Select next sample to update */
        i = iVals[k]-1;
        
        alphaOld = alpha[i];
        if(alphaOld == 0 || alphaOld == 1)
            alpha[i] = 0.5;
        LB = 0.0;
        UB = 1.0;
        c = (alphaOld/lambda)*yxxy[i];
        if (sparse) {
            for(j = jc[i];j < jc[i+1];j++) {
                c += w[ir[j]]*yXt[j];
            }
        }
        else {
            for(j = 0;j < nVars; j++) {
                c += w[j]*yXt[j + nVars*i];
            }
        }
        d = yxxy[i]/lambda;
        gi = log(alpha[i]) - log(1-alpha[i]) - c + alpha[i]*d;
        while (fabs(gi) > 1e-8 && UB-LB > 1e-15) {
            Hi = 1/alpha[i] + 1/(1-alpha[i]) + d;
            alpha[i] -= gi/Hi;
            if (alpha[i] <= LB || alpha[i] >= UB)
                alpha[i] = (LB+UB)/2;
            gi = log(alpha[i]) - log(1-alpha[i]) - c + alpha[i]*d;
            if (gi > 0)
                UB = alpha[i];
            else
                LB = alpha[i];
        }
        
        if (sparse) {
            for(j = jc[i];j < jc[i+1];j++) {
                w[ir[j]] += yXt[j]*(alphaOld - alpha[i])/lambda;
            }
        }
        else {
            for(j = 0;j < nVars;j++) {
                w[j] += yXt[j + nVars*i]*(alphaOld - alpha[i])/lambda;
            }
        }
        
    }
    
    if (nrhs < 5)
        mxFree(w);
    if (nrhs < 6)
        mxFree(yxxy);
    
}
