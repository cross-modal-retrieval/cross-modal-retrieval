#include <math.h>
#include "mex.h"

/* 
 * ASGD_logistic(w,Xt,y,lambda,stepSizes,iVals,average);
 w(p,1) - updated in place
 Xt(p,n) - real, can be sparse
 y(n,1) - {-1,1}
 lambda - scalar regularization param
 stepSizes(maxIter,1) - sequence of step sizes
 iVals(maxIter,1) - sequence of examples to choose

 If an output argument is requested, it returns the average of the iterates,
 */


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int k,nSamples,maxIter,sparse=0,*iVals,average=0,useScaling=1,loss;
    long i,j,nVars;
    
    mwIndex *jc,*ir;
    
    double *w, *Xt, *y, lambda, *stepSizes, innerProd, alpha,sig,c=1,cA=1,tau=0,*wAvg,*averageWeights,weightedAverage=0;
    
    /* Input */
    
    if (nrhs < 6)
        mexErrMsgTxt("Exactly 6 arguments are needed: {w,Xy,y,lambda,stepSizes,iVals}");
    
    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    stepSizes = mxGetPr(prhs[4]);
    iVals = (int*)mxGetPr(prhs[5]);
    
    if (!mxIsClass(prhs[5],"int32"))
        mexErrMsgTxt("iVals must be int32");
    
    /* Compute Sizes */
    nVars = mxGetM(prhs[1]);
    nSamples = mxGetN(prhs[1]);
    maxIter = mxGetM(prhs[4]);
    
    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[2]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");
    if (maxIter != mxGetM(prhs[5]))
        mexErrMsgTxt("iVals and stepSizes must have the same number of rows");
    if (nrhs > 7 && maxIter != mxGetM(prhs[7]))
        mexErrMsgTxt("iVals and averageWeights must have the same number of rows");
    
    if (mxIsSparse(prhs[1])) {
        sparse = 1;
        jc = mxGetJc(prhs[1]);
        ir = mxGetIr(prhs[1]);
    }
    
    
    if (nlhs > 0) {
        average = 1;
        plhs[0] = mxCreateDoubleMatrix(nVars,1,mxREAL);
        wAvg = mxGetPr(plhs[0]);
    }
    
    for(k=0;k<maxIter;k++)
    {
        /* Select next training example */
        i = iVals[k]-1;
        
        /* Compute Inner Product of Parameters with Features */
        innerProd = 0;
        if(sparse) {
            for(j=jc[i];j<jc[i+1];j++)
                innerProd += w[ir[j]]*Xt[j];
            if (useScaling)
                innerProd *= c;
            
        }
        else {
            for(j=0;j<nVars;j++)
                innerProd += w[j]*Xt[j + nVars*i];
        }
        sig = -y[i]/(1+exp(y[i]*innerProd));
        
        /* Compute step size */
        alpha = stepSizes[k];
        
        /* Update parameters */
        if (sparse) {
            if (useScaling) {
                if (alpha*lambda != 1) {
                    c *= 1-alpha*lambda;
                }
                else {
                    c = 1;
                    for(j=0;j<nVars;j++) {
                        w[j] = 0;
                    }
                }
                for(j=jc[i];j<jc[i+1];j++)
                    w[ir[j]] -= alpha*Xt[j]*sig/c;
            }
            else {
                for(j=0;j<nVars;j++)
                    w[j] *= (1-alpha*lambda);
                for(j=jc[i];j<jc[i+1];j++)
                    w[ir[j]] -= alpha*Xt[j]*sig;
            }
            
        }
        else {
            for(j=0;j<nVars;j++)
                w[j] *= 1-alpha*lambda;
            for(j=0;j<nVars;j++)
                w[j] -= alpha*Xt[j + i*nVars]*sig;
        }
        
        /* Average */
        if(average) {
            if (sparse && useScaling) {
                if (k > 0)
                    cA *= (double)k/(double)(k+1);
                for(j=jc[i];j<jc[i+1];j++)
                    wAvg[ir[j]] += tau*alpha*Xt[j]*sig/c;
                tau += (c/cA)/(double)(k+1);
                      
                /* Slow way
                 * 
                 * for(j=0;j<nVars;j++)
                    wAvg[j] += c*w[j];*/
            }
            else {
                for(j=0;j<nVars;j++)
                    wAvg[j] += w[j];
            }
        }
        
    }

    if(sparse && useScaling) {
        for(j=0;j<nVars;j++)
            w[j] *= c;
    }
        
    if(average) {
        if (sparse && useScaling) {
            for(j=0;j<nVars;j++)
                wAvg[j] = ((tau/c)*w[j] + wAvg[j])*cA;
        }
        else
        {
        for(j=0;j<nVars;j++)
            wAvg[j] /= maxIter;
        }
    }
    
}
