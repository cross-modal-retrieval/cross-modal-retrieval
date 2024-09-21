
int getMaxState(int *nStates,int nNodes)
{
   int n, maxState=0;
   
   for(n = 0; n < nNodes; n++)
   {
      if(nStates[n] > maxState)
         maxState = nStates[n];
   }
   return maxState;
}

double absDif(double a, double b)
 {
     if (a > b)
     {
         return a-b;
     }
     else
     {
         return b-a;
     }
 }
 
 
