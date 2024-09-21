#include <math.h>
#include "mex.h"

/* 
 * SGD_logistic(w,Xt,y,lambda,stepSizes,iVals,maxNorm,average);
 w(p,1) - updated in place
 Xt(p,n) - real, can be sparse
 y(n,1) - {-1,1}
 lambda - scalar regularization param
 stepSizes(maxIter,1) - sequence of step sizes
 iVals(maxIter,1) - sequence of examples to choose
 maxNorm - radius of L1-norm ball to project onto (set to 0 for no projection)
 averageWeights - if doing averaging, use these weights instead of 1/maxIter

 If an output argument is requested, it returns the average of the iterates,
 but note that this slow for sparse matrices since it does not use the sparsity (use ASG_logistic instead)
 */


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int k,nSamples,maxIter,sparse=0,*iVals,average=0,useScaling=1,loss;
    long i,j,nVars;
    
    mwIndex *jc,*ir;
    
    double *w, *Xt, *y, lambda, *stepSizes, innerProd, alpha,sig,c=1,maxNorm=0,nrmW,*wAvg,*averageWeights,weightedAverage=0;
    
    /* Input */
    
    if (nrhs < 6)
        mexErrMsgTxt("At least 6 arguments are needed: {w,Xy,y,lambda,stepSizes,iVals,maxNorm}");
    
    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    stepSizes = mxGetPr(prhs[4]);
    iVals = (int*)mxGetPr(prhs[5]);
    if (nrhs > 6)
        maxNorm = mxGetScalar(prhs[6]);
    if (nrhs > 7) {
        weightedAverage = 1;
        averageWeights = mxGetPr(prhs[7]);
    }
        
    
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
    
    if (maxNorm!=0 && sparse && useScaling) {
        nrmW = 0;
        for(j=0;j<nVars;j++) {
            nrmW += w[j]*w[j];
        }
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
        
        /* Subtract old values from norm */
        if (maxNorm!=0 && sparse && useScaling) {
            for(j=jc[i];j<jc[i+1];j++)
                nrmW -= w[ir[j]]*w[ir[j]];
        }
        
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
        
        /* Project */
        if(maxNorm!=0) {
            if (sparse && useScaling) {
                if (alpha*lambda==1)
                    nrmW = 0;
                for(j=jc[i];j<jc[i+1];j++)
                    nrmW += w[ir[j]]*w[ir[j]]; /* Add new values to norm */
                
                if (c > 0 && c*sqrt(nrmW) > maxNorm)
                    c = maxNorm/sqrt(nrmW);
                else if (c < 0 && -c*sqrt(nrmW) > maxNorm)
                    c = -maxNorm/sqrt(nrmW);
            }
            else {
                nrmW = 0;
                for(j=0;j<nVars;j++)
                        nrmW += w[j]*w[j];
                if(sqrt(nrmW) > maxNorm) {
                    for(j=0;j<nVars;j++)
                        w[j] *= maxNorm/sqrt(nrmW);
                }
            }
        }
        
        /* Average */
        if(average) {
            if (!weightedAverage) {
                if (sparse && useScaling) {
                    for(j=0;j<nVars;j++)
                        wAvg[j] += c*w[j];
                }
                else {
                    for(j=0;j<nVars;j++)
                        wAvg[j] += w[j];
                }
            }
            else if (averageWeights[k]!=0) {
                if (sparse && useScaling) {
                    for(j=0;j<nVars;j++)
                        wAvg[j] += c*w[j]*averageWeights[k];
                }
                else {
                    for(j=0;j<nVars;j++)
                        wAvg[j] += w[j]*averageWeights[k];
                }
            }
        }
        
        /* Re-normalize the parameter vector if it has gone numerically crazy */
        if(c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100)) {
            printf("Re-normalizing\n");
            for(j=0;j<nVars;j++) {
                w[j] = c*w[j];
            }
            c = 1;
        }
        
    }
    
    if(average && !weightedAverage) {
        for(j=0;j<nVars;j++)
            wAvg[j] /= maxIter;
    }
    if(sparse && useScaling) {
        for(j=0;j<nVars;j++)
            w[j] *= c;
    }
    
}
