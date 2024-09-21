#include <math.h>
#include "mex.h"

/*
SAG_logistic(w,Xt,y,lambda,alpha,iVals,d,g,covered);
% w(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {-1,1}
% lambda - scalar regularization param
% stepSize - scalar constant step size
% iVals(maxIter,1) - sequence of examples to choose
%
% The below are updated in place and are needed for restarting the algorithm
% d(p,1) - initial approximation of average gradient (should be sum of previous gradients)
% g(n,1) - previous derivatives of loss
% covered(n,1) - whether the example has been visited
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int k,nSamples,maxIter,sparse=0,*iVals,useScaling=1,*covered,useLazy=1,*lastVisited;
    long i,j,nVars;
    
    mwIndex *jc,*ir;
    
    double *w, *Xt, *y, lambda, alpha, innerProd, sig,c=1,*g,*d,nCovered=0,*cumSum;
    
    if (nrhs != 9)
        mexErrMsgTxt("Function needs nine arguments: {w,Xt,y,lambda,alpha,iVals,d,g,covered}");
    
    /* Input */
    
    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    alpha = mxGetScalar(prhs[4]);
    iVals = (int*)mxGetPr(prhs[5]);
    if (!mxIsClass(prhs[5],"int32"))
        mexErrMsgTxt("iVals must be int32");
    d = mxGetPr(prhs[6]);
    g = mxGetPr(prhs[7]);
    covered = (int*)mxGetPr(prhs[8]);
    
    /* Compute Sizes */
    nVars = mxGetM(prhs[1]);
    nSamples = mxGetN(prhs[1]);
    maxIter = mxGetM(prhs[5]);
    
    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[2]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");
    if (nVars != mxGetM(prhs[6]))
        mexErrMsgTxt("w and d must have the same number of rows");
    if (nSamples != mxGetM(prhs[7]))
        mexErrMsgTxt("w and g must have the same number of rows");
    if (nSamples != mxGetM(prhs[8]))
        mexErrMsgTxt("covered and y must hvae the same number of rows");
        
    if (mxIsSparse(prhs[1])) {
        sparse = 1;
        jc = mxGetJc(prhs[1]);
        ir = mxGetIr(prhs[1]);
    }
    else {
        useScaling = 0;
        useLazy = 0;
    }
    
    if (sparse && alpha*lambda==1)
        mexErrMsgTxt("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
    
    /* Allocate memory needed for lazy updates */
    if (useLazy) {
        lastVisited = mxCalloc(nVars,sizeof(int));
        cumSum = mxCalloc(maxIter,sizeof(double));
        
        /*for(j=0;j<nVars;j++)
            lastVisited[j] = -1;*/
    }
    
    for(i=0;i<nSamples;i++) {
        if (covered[i]!=0)
            nCovered++;
    }
    
    for(k=0;k<maxIter;k++)
    {
        /* Select next training example */
        i = iVals[k]-1;
        
        /* Compute current values of needed parameters */
        if (useLazy && k > 0) {
            for(j=jc[i];j<jc[i+1];j++) {
                if (lastVisited[ir[j]]==0) {
                    w[ir[j]] -= d[ir[j]]*cumSum[k-1];
                }
                else {
                    w[ir[j]] -= d[ir[j]]*(cumSum[k-1]-cumSum[lastVisited[ir[j]]-1]);
                }
                lastVisited[ir[j]] = k;
            }
        }
        
        /* Compute derivative of loss */
        innerProd = 0;
        if (sparse) {
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
        
        /* Update direction */
        if (sparse) {
            for(j=jc[i];j<jc[i+1];j++)
                d[ir[j]] += Xt[j]*(sig - g[i]);
        }
        else {
            for(j=0;j<nVars;j++)
                d[j] += Xt[j + nVars*i]*(sig - g[i]);
        }
        
        /* Store derivative of loss */
        g[i] = sig;
            
        /* Update the number of examples that we have seen */
        if (covered[i]==0) {
            covered[i]=1;
            nCovered++;
        }
        
        /* Update parameters */
        if (useScaling)
        {
            c *= 1-alpha*lambda;
            
            if (useLazy) {
                if (k==0)
                    cumSum[0] = alpha/(c*nCovered);
                else
                    cumSum[k] = cumSum[k-1] + alpha/(c*nCovered);
            }
            else {
                for(j=0;j<nVars;j++)
                    w[j] -= alpha*d[j]/(c*nCovered);
            }
        }
        else {
            for(j=0;j<nVars;j++) {
                w[j] *= 1-alpha*lambda;
            }
            
            for(j=0;j<nVars;j++)
                w[j] -= alpha*d[j]/nCovered;
        }
        
         /* Re-normalize the parameter vector if it has gone numerically crazy */
        if(c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100)) {
            printf("Re-normalizing\n");
            
            if (useLazy) {
                for(j=0;j<nVars;j++) {
                    if (lastVisited[j]==0)
                        w[j] -= d[j]*cumSum[k];
                    else
                        w[j] -= d[j]*(cumSum[k]-cumSum[lastVisited[j]-1]);
                    lastVisited[j] = k+1;
                }
                cumSum[k] = 0;
            }
            for(j=0;j<nVars;j++) {
                w[j] = c*w[j];
            }
            c = 1;
        }
        
    }
    
    if (useLazy) {
        for(j=0;j<nVars;j++) {
            if (lastVisited[j]==0) {
                w[j] -= d[j]*cumSum[maxIter-1];
            }
            else
            {
                w[j] -= d[j]*(cumSum[maxIter-1]-cumSum[lastVisited[j]-1]);
            }
        }
    }
    
    if(useScaling) {
        for(j=0;j<nVars;j++)
            w[j] *= c;
    }
        
    if(useLazy) {
        mxFree(lastVisited);
        mxFree(cumSum);
    }
}
