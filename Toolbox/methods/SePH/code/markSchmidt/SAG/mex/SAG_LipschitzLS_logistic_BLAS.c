#include <math.h>
#include "mex.h"
#include "blas.h"

/*
SAG_LipschitzLS_logistic(w,Xt,y,lambda,Lmax,Li,randVals,d,g,covered[,increasing,xtx]);
% w(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {-1,1}
% lambda - scalar regularization param
% Lmax - initial approximation of global Lipschitz constant
% Li(n,1) - initial approximations of individual Lipschitz constants
% randVals(maxIter,2) - sequence of random values for the algorithm to use
%
% The below are updated in place and are needed for restarting the algorithm
% d(p,1) - initial approximation of average gradient (should be sum of previous gradients)
% g(n,1) - previous derivatives of loss
% covered(n,1) - whether the example has been visited
%
% increasing - default is 1 to allow the Lipscthiz constants to increase, set to 0 to only allow them to decrease
% xtx - squared norms of features
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int k,ind,nSamples,maxIter,sparse=0,*covered,*lastVisited,increasing=0,temp,
            nextpow2,levelMax,nLevels,level;
    long i,j,nVars,one=1;
    
    mwIndex *jc,*ir;
    
    double *w, *Xt, *y, lambda, *Li, alpha, innerProd, sig,c=1,*g,*d,nCovered=0,*cumSum,fi,fi_new,gg,precision,
            *randVals,*Lmax,Lmean,Li_old,*nDescendants,*unCoveredMatrix,*LiMatrix,offset,u,z,Z,wtx,*xtx,scaling;
    
    if (nrhs < 10)
        mexErrMsgTxt("Function needs nine arguments: {w,Xt,y,lambda,Lmax,Li,randVals,d,g,covered[,increasing,xtx]}");
    
    /* Input */
    
    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    Lmax = mxGetPr(prhs[4]);
    Li = mxGetPr(prhs[5]);
    randVals = mxGetPr(prhs[6]);
    d = mxGetPr(prhs[7]);
    g = mxGetPr(prhs[8]);
    covered = (int*)mxGetPr(prhs[9]);
    
    if (nrhs >= 11) {
        increasing = (int)mxGetScalar(prhs[10]);
        if (!mxIsClass(prhs[10],"int32"))
            mexErrMsgTxt("increasing must be int32");
    }    
    
    /* Compute Sizes */
    nVars = mxGetM(prhs[1]);
    nSamples = mxGetN(prhs[1]);
    maxIter = mxGetM(prhs[6]);
    precision = 1.490116119384765625e-8;
    
    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[2]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");
    if (nVars != mxGetM(prhs[7]))
        mexErrMsgTxt("w and d must have the same number of rows");
    if (nSamples != mxGetM(prhs[8]))
        mexErrMsgTxt("w and g must have the same number of rows");
    if (nSamples != mxGetM(prhs[9]))
        mexErrMsgTxt("covered and y must hvae the same number of rows");
        
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
    }
    
    /* Compute mean of covered variables */
    Lmean = 0;
    for(i=0;i<nSamples;i++) {
        if (covered[i]!=0) {
            nCovered++;
            Lmean += Li[i];
        }
    }
    if(nCovered > 0)
        Lmean /= nCovered;
    
    if (nrhs >= 12) {
        xtx = mxGetPr(prhs[11]);
        if (nSamples != mxGetM(prhs[11]))
            mexErrMsgTxt("covered and xtx must have the same number or rows");
    }
    else {
        xtx = mxCalloc(nSamples,sizeof(double));
        for(i = 0; i < nSamples;i++) {
            xtx[i] = 0;
            if (sparse) {
                for(j=jc[i];j<jc[i+1];j++)
                    xtx[i] += Xt[j]*Xt[j];
            }
            else
                xtx[i] = ddot(&nVars,&Xt[i*nVars],&one,&Xt[i*nVars],&one);
        }
    }
    
    /* Do the O(n log n) initialization of the data structures will allow sampling in O(log(n)) time */
    nextpow2 = pow(2,ceil(log2(nSamples)/log2(2)));
    nLevels = 1+(int)ceil(log2(nSamples));
    /*printf("next power of 2 is: %d\n",nextpow2);
    printf("nLevels = %d\n",nLevels);*/
    
    nDescendants = mxCalloc(nextpow2*nLevels,sizeof(double)); /* Counts number of descendents in tree */
    unCoveredMatrix = mxCalloc(nextpow2*nLevels,sizeof(double)); /* Counts number of descenents that are still uncovered */
    LiMatrix = mxCalloc(nextpow2*nLevels,sizeof(double)); /* Sums Lipschitz constant of loss over descendants */
    
    for(i=0;i<nSamples;i++) {
        nDescendants[i] = 1;
        if (covered[i]) 
            LiMatrix[i] = Li[i];
        else
            unCoveredMatrix[i] = 1;
    }
    levelMax = nextpow2;
    for (level=1;level<nLevels;level++) {
        levelMax = levelMax/2;
        for(i=0;i<levelMax;i++) {
            nDescendants[i + nextpow2*level] = nDescendants[2*i + nextpow2*(level-1)] + nDescendants[2*i+1 + nextpow2*(level-1)];
            LiMatrix[i + nextpow2*level] = LiMatrix[2*i + nextpow2*(level-1)] + LiMatrix[2*i+1 + nextpow2*(level-1)];
            unCoveredMatrix[i + nextpow2*level] = unCoveredMatrix[2*i + nextpow2*(level-1)] + unCoveredMatrix[2*i+1 + nextpow2*(level-1)];
        }
    }
    
    /*  
     for(ind=0;ind<nextpow2;ind++) {
        for(j=0;j<nLevels;j++) {
            printf("%f ",unCoveredMatrix[ind + nextpow2*j]);
        }
        printf("\n");
    }
     */
    
    for(k=0;k<maxIter;k++)
    {
        /* Select next training example */
        offset = 0;
        i = 0;
        u = randVals[k+maxIter];
        if(randVals[k] < (double)(nSamples-nCovered)/(double)nSamples) {
            /* Sample fron uncovered guys */
            Z = unCoveredMatrix[nextpow2*(nLevels-1)];
            for(level=nLevels-1;level>=0;level--) {
                z = offset + unCoveredMatrix[2*i + nextpow2*level];
                if(u < z/Z)
                    i = 2*i;
                else {
                    offset = z;
                    i = 2*i+1;
                }
            }
        }
        else {
            /* Sample from covered guys according to estimate of Lipschitz constant */
            Z = LiMatrix[nextpow2*(nLevels-1)] + (Lmean + 2*lambda)*(nDescendants[nextpow2*(nLevels-1)] - unCoveredMatrix[nextpow2*(nLevels-1)]);
            for(level=nLevels-1;level>=0;level--) {
                z = offset + LiMatrix[2*i + nextpow2*level] + (Lmean + 2*lambda)*(nDescendants[2*i + nextpow2*level] - unCoveredMatrix[2*i + nextpow2*level]);
                if(u < z/Z)
                    i = 2*i;
                else {
                    offset = z;
                    i = 2*i+1;
                }
            }
            /*printf("i = %d\n",i);*/
        }
        
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
        innerProd = 0;
        if (sparse) {
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
            
        /* Line-search for Li */
        Li_old = Li[i];
        if(increasing && covered[i])
            Li[i] /= 2;
        fi = log(1 + exp(-y[i]*innerProd));
        /* Compute f_new as the function value obtained by taking 
         * a step size of 1/Li in the gradient direction */
         wtx = 0;
        if (sparse) {
            for(j=jc[i];j<jc[i+1];j++)
                wtx += c*w[ir[j]]*Xt[j];
        }
        else
            wtx = ddot(&nVars,&Xt[i*nVars],&one,w,&one);
        gg = sig*sig*xtx[i];
        innerProd = wtx - xtx[i]*sig/Li[i];
        fi_new = log(1 + exp(-y[i]*innerProd));
        /*printf("fi = %e, fi_new = %e, gg = %e\n",fi,fi_new,gg);*/
        while (gg > precision && fi_new > fi - gg/(2*(Li[i]))) {
            /*printf("Lipschitz Backtracking (k = %d, fi = %e, fi_new = %e, 1/Li = %e)\n",k+1,fi,fi_new,1/(Li[i]));*/
            Li[i] *= 2;
            innerProd = wtx - xtx[i]*sig/Li[i];
            fi_new = log(1 + exp(-y[i]*innerProd));
            
        }

        if(Li[i] > *Lmax)
            *Lmax = Li[i];
        
        /* Update the number of examples that we have seen */
        if (covered[i]==0) {
            covered[i]=1;
            nCovered++;
            Lmean = Lmean*((double)(nCovered-1)/(double)nCovered) + Li[i]/(double)nCovered;
            
            /* Update unCoveredMatrix so we don't sample this guy when looking for a new guy */
            ind = i;
            for(level=0;level<nLevels;level++)
            {
                unCoveredMatrix[ind + nextpow2*level] -= 1;
                ind = ind/2;
            }
            /* Update LiMatrix so we sample this guy proportional to its Lipschitz constant*/
            ind = i;
            for(level=0;level<nLevels;level++)
            {
                LiMatrix[ind + nextpow2*level] += Li[i];
                ind = ind/2;
            }
        }
        else if (Li[i] != Li_old) {
            Lmean = Lmean + (Li[i] - Li_old)/(double)nCovered;
            /* Update LiMatrix with the new estimate of the Lipscitz constant */
            ind = i;
            for(level=0;level<nLevels;level++)
            {
                LiMatrix[ind + nextpow2*level] += (Li[i] - Li_old);
                ind = ind/2;
            }
        }
        
        /*for(ind=0;ind<nextpow2;ind++) {
            for(j=0;j<nLevels;j++) {
                printf("%f ",LiMatrix[ind + nextpow2*j]);
            }
            printf("\n");
        }
        
        /* Compute step size */
        alpha = ((double)(nSamples-nCovered)/(double)nSamples)/(*Lmax + lambda) + ((double)nCovered/(double)nSamples)*(1/(2*(*Lmax + lambda)) + 1/(2*(Lmean + lambda)));

        
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
                
        /* Decrease value of max Lipschitz constant */
        if (increasing)
            *Lmax *= pow(2.0,-1.0/nSamples);
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
    mxFree(nDescendants);
    mxFree(unCoveredMatrix);
    mxFree(LiMatrix);
    if (nrhs < 12)
        mxFree(xtx);
}
