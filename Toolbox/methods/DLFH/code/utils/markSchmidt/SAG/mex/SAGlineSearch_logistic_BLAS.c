#include <math.h>
#include "mex.h"
#include "blas.h"

/*
SAGlineSearch_logistic(w,Xt,y,lambda,alpha,iVals,d,g,covered[,stepSizeType,xtx]);
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
%
% stepSizeType - default is 1 to use 1/L, setting to 2 uses 2/(L + n*mu)
% xtx - squared norms of features
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int k,nSamples,maxIter,sparse=0,*iVals,*covered,*lastVisited,stepSizeType=1,temp;
    long i,j,nVars,one=1;
    
    mwIndex *jc,*ir;
    
    double *w, *Xt, *y, lambda, *Li, alpha, innerProd, sig,c=1,*g,*d,nCovered=0,*cumSum,fi,fi_new,gg,precision,scaling,wtx,*xtx;
    
    if (nrhs < 9)
        mexErrMsgTxt("Function needs nine arguments: {w,Xt,y,lambda,alpha,iVals,d,g,covered[,stepSizeType,xtx]}");
    
    /* Input */
    
    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    Li = mxGetPr(prhs[4]);
    iVals = (int*)mxGetPr(prhs[5]);
    if (!mxIsClass(prhs[5],"int32"))
        mexErrMsgTxt("iVals must be int32");
    d = mxGetPr(prhs[6]);
    g = mxGetPr(prhs[7]);
    covered = (int*)mxGetPr(prhs[8]);
    
    if (nrhs >= 10) {
        stepSizeType = (int)mxGetScalar(prhs[9]);
        if (!mxIsClass(prhs[9],"int32"))
            mexErrMsgTxt("stepSizeType must be int32");
    }    
    
    /* Compute Sizes */
    nVars = mxGetM(prhs[1]);
    nSamples = mxGetN(prhs[1]);
    maxIter = mxGetM(prhs[5]);
    precision = 1.490116119384765625e-8;
    
    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[2]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");
    if (nVars != mxGetM(prhs[6]))
        mexErrMsgTxt("w and d must have the same number of rows");
    if (nSamples != mxGetM(prhs[7]))
        mexErrMsgTxt("w and g must have the same number of rows");
    if (nSamples != mxGetM(prhs[8]))
        mexErrMsgTxt("covered and y must have the same number of rows");
        
    if (mxIsSparse(prhs[1])) {
        sparse = 1;
        jc = mxGetJc(prhs[1]);
        ir = mxGetIr(prhs[1]);
    }
    
    if (sparse && alpha*lambda==1)
        mexErrMsgTxt("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
    
    /* Allocate memory needed for lazy updates */
    if (sparse) {
        lastVisited = mxCalloc(nVars,sizeof(double));
        cumSum = mxCalloc(maxIter,sizeof(double));
        
        /*for(j=0;j<nVars;j++)
            lastVisited[j] = -1;*/
    }
    
    for(i=0;i<nSamples;i++) {
        if (covered[i]!=0)
            nCovered++;
    }
    
    if (nrhs >= 11) {
        xtx = mxGetPr(prhs[10]);
        if (nSamples != mxGetM(prhs[10]))
            mexErrMsgTxt("covered and xtx must have the same number or rows");
    }
    else {
        xtx = mxCalloc(nSamples,sizeof(double));
        for(i = 0; i < nSamples;i++) {
            if (sparse) {
                xtx[i] = 0;
                for(j=jc[i];j<jc[i+1];j++)
                    xtx[i] += Xt[j]*Xt[j];
            }
            else
                xtx[i] = ddot(&nVars,&Xt[i*nVars],&one,&Xt[i*nVars],&one);
        }
    }
    
    for(k=0;k<maxIter;k++)
    {
        /* Select next training example */
        i = iVals[k]-1;
        
        /* Compute current values of needed parameters */
        if (sparse && k > 0) {
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
        if (sparse) {
            innerProd = 0;
            for(j=jc[i];j<jc[i+1];j++)
                innerProd += w[ir[j]]*Xt[j];
            innerProd *= c;
        }
        else
            innerProd = ddot(&nVars,w,&one,&Xt[nVars*i],&one);
        
        sig = -y[i]/(1+exp(y[i]*innerProd));
        
        /* Update direction */
        if (sparse) {
            for(j=jc[i];j<jc[i+1];j++)
                d[ir[j]] += Xt[j]*(sig - g[i]);
        }
        else {
            scaling = sig-g[i];
            daxpy(&nVars,&scaling,&Xt[i*nVars],&one,d,&one);
        }
        
        /* Store derivative of loss */
        g[i] = sig;
            
        /* Update the number of examples that we have seen */
        if (covered[i]==0) {
            covered[i]=1;
            nCovered++;
        }
        
        /* Line-search for Li */
        fi = log(1 + exp(-y[i]*innerProd));
        /* Compute f_new as the function value obtained by taking 
         * a step size of 1/Li in the gradient direction */
        if (sparse) {
            wtx = 0;
            for(j=jc[i];j<jc[i+1];j++) {
                wtx += c*w[ir[j]]*Xt[j];
            }
        }
        else
            wtx = ddot(&nVars,&Xt[i*nVars],&one,w,&one);
        gg = sig*sig*xtx[i];
        innerProd = wtx - xtx[i]*sig/(*Li);
        fi_new = log(1 + exp(-y[i]*innerProd));
        /*printf("fi = %e, fi_new = %e, gg = %e\n",fi,fi_new,gg);*/
        while (gg > precision && fi_new > fi - gg/(2*(*Li))) {
            /*printf("Lipschitz Backtracking (k = %d, fi = %e, fi_new = %e, 1/Li = %e)\n",k+1,fi,fi_new,1/(*Li));*/
            *Li *= 2;
            innerProd = wtx - xtx[i]*sig/(*Li);
            fi_new = log(1 + exp(-y[i]*innerProd));
            
        }

        /* Compute step size */
        if (stepSizeType == 1)
            alpha = 1/(*Li + lambda);
        else
            alpha = 2/(*Li + (nSamples+1)*lambda);

        
        /* Update parameters */
        if (sparse)
        {
            c *= 1-alpha*lambda;
            
            if (k==0)
                cumSum[0] = alpha/(c*nCovered);
            else
                cumSum[k] = cumSum[k-1] + alpha/(c*nCovered);
        }
        else {
            scaling = 1-alpha*lambda;
            dscal(&nVars,&scaling,w,&one);
            scaling = -alpha/nCovered;
            daxpy(&nVars,&scaling,d,&one,w,&one);
        }
        
        /* Decrease value of Lipschitz constant */
        *Li *= pow(2.0,-1.0/nSamples);
                
    }
    
    if (sparse) {
        for(j=0;j<nVars;j++) {
            if (lastVisited[j]==0) {
                w[j] -= d[j]*cumSum[maxIter-1];
            }
            else
            {
                w[j] -= d[j]*(cumSum[maxIter-1]-cumSum[lastVisited[j]-1]);
            }
        }
        scaling = c;
        dscal(&nVars,&scaling,w,&one);
        mxFree(lastVisited);
        mxFree(cumSum);
    }
    if (nrhs < 11)
        mxFree(xtx);
}
