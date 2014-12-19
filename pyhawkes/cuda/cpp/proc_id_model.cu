#include <cuda.h>
#include <math_constants.h>

#define OP_SUM  0
#define OP_MULT 1

const int B = %(B)s;               // blockDim.x

/**
 * Helper function to sum across a block.
 * Assume pS_data is already in shared memory
 * Only the first thread returns a value in pSum
 */
__device__ void reduceBlock( float pSdata[B], float* pSum, int op )
{
   int idx = threadIdx.x * blockDim.y + threadIdx.y;

   // Sync all threads across the block
   __syncthreads();

   // Calculate the minimum value by doing a reduction
   int half = (blockDim.x*blockDim.y) / 2;
   if( idx < half )
   {
       while( half > 0 )
       {
           if(idx < half)
           {
               switch(op)
               {
                   case OP_SUM:
                       pSdata[idx] = pSdata[idx] + pSdata[idx + half];
                       break;
                   case OP_MULT:
                       pSdata[idx] = pSdata[idx] * pSdata[idx + half];
                       break;
                   default:
                       // default to the identity
                       // TODO: throw error?
                       pSdata[idx] = pSdata[idx];
                       break;
               }
           }
           half = half / 2;
           __syncthreads();
       }
   }

   // Store the minimum value back to global memory
   if (idx == 0)
   {
       pSum[0] = pSdata[0];
   }
}

/**
 * Compute the mean of each component in a Gaussian mixture model
 * ((N/B)xK) grid of (Bx1x1) blocks
 */
__global__ void computeXSum(int N,
		                    int* pC,
		                    int D,
		                    int d,
		                    float* pX,
		                    float* pXMeanSum)
{
	int k = blockIdx.y;
	int x = threadIdx.x;
	int n = blockIdx.x * blockDim.x + x;
	
	__shared__ float xsum[B];
	xsum[x] = 0.0;
	
	
	if (n<N)
	{
		if (pC[n]==k)
		{
			xsum[x] = pX[d*N+n];
		}
	}
	
	// Sum xsum for this block
	// the output array is KxGridDim.x in size
	float blockXsum = 0.0;
	reduceBlock(xsum, &blockXsum, OP_SUM);
	
	if (x==0)
	{
		pXMeanSum[k*gridDim.x+blockIdx.x] = blockXsum;
	}
	
}

/**
 * Compute the mean of each component in a Gaussian mixture model
 * ((N/B)xK) grid of (Bx1x1) blocks
 * 
 * pXvarsum is a KxDxD array
 */
__global__ void computeXVarSum(int N,
		                       int D,
							   int* pC,
							   int d1,
							   int d2,
							   float* pX,
							   float* pXmean,
							   float* pXCovarSum)
{
	int k = blockIdx.y;
	int x = threadIdx.x;
	int n = blockIdx.x * blockDim.x + x;
	
	__shared__ float xvarsum[B];
	xvarsum[x] = 0.0;
	
	float xmean1 = pXmean[k*D+d1];
	float xmean2 = pXmean[k*D+d2];
	
	if (n<N)
	{
		if (pC[n]==k)
		{
			xvarsum[x] = (pX[d1*N+n]-xmean1)*(pX[d2*N+n]-xmean2);
		}
	}
	
	// Sum xsum for this block
	// the output array is KxGridDim.x in size
	float blockXvarsum = 0.0;
	reduceBlock(xvarsum, &blockXvarsum, OP_SUM);
	
	if (x==0)
	{
		pXCovarSum[k*gridDim.x+blockIdx.x] = blockXvarsum;
	}
	
}

/**
 * Compute the per-spike contributions to the probability of the n-th spike
 * occuring on process k. Spike n's process identity affects the parenting 
 * spike Z[n] as well as any child spikes n' such that Z[n']=n. 
 */
__global__ void computePerSpikePrCn(int n,
		                            int N,
		                            int K,
		                            int D,
		                            float* pX,
		                            float* pMuX,
		                            float* pLamX,
		                            int* pZ,
		                            int* pC,
		                            bool* pA,
		                            float* pW,
		                            float* pLam,
		                            float* pPrCn)
{
	int cn = blockIdx.y;
	int x = threadIdx.x;
	int m = blockIdx.x * blockDim.x + x; // this thread represents the m-th spike
	
	__shared__ float logprcn[B];
	logprcn[x] = 0.0;
	
	if (m<N)
	{
		// Get the m-th spike's process identity
		int cm = pC[m];
		
		if (m<n && pZ[n]==m)
		{
			// if the m-th spike is the parent of spike n, calculate its contribution
			logprcn[x] += logf(pA[cm*K+cn]) + logf(pW[cm*K+cn]);
		}
		else if (m==n)
		{
			// if m==n then calculate the prior pr given mu and the pr if it is parented
			// by the background
			if (pZ[n]==-1)
			{
				// TODO: UPDATE PYTHON CODE FOR NON-HOMOGENEOUS BACKGROUND RATES
				logprcn[x] += logf(pLam[cn*N+n]);
			}
			
			// Calculate the prior probability given mu and sigma
			// This involves a vector-matrix-vector multiply for the Gaussian logprob
			for (int d1=0; d1<D; d1++)
			{
				for (int d2=0; d2<D; d2++)
				{
					float xn1 = pX[d1*N+n];
					float xn2 = pX[d2*N+n];
					logprcn[x] += -1*(xn1-pMuX[cn*D+d1])*pLamX[cn*D*D+d1*D+d2]*(xn2-pMuX[cn*D+d2]);
				}
			}
		}
		else if (m>n && pZ[m]==n)
		{
			// if the m-th spike is parented by spike n calculate its contribution
			logprcn[x] += logf(pA[cn*K+cm]) + logf(pW[cn*K+cm]);
		}
		
	}
	
	// Sum the result for each block
	// the output array is KxGridDim.x in size
	float blockLogPrCnSum = 0.0;
	reduceBlock(logprcn, &blockLogPrCnSum, OP_SUM);
	
	if (x==0)
	{
		pPrCn[cn*gridDim.x+blockIdx.x] = blockLogPrCnSum;
	}
}

/**
 * Compute the log Q-ratio for accepting a MH proposal made by the 
 * MetaProcessId model. The proposal is to change all spikes affiliated with
 * underlying process k to a new meta-process m, while simultaneously updating 
 * the parent assignments Z. The log probability of accepting such a proposal is
 * proportional to a sum over terms for every spike that either occurs on the
 * underlying process k, or may have a parent spike on underlying process k.
 */
__global__ void computeLogQratio(int k,
							 	int m,
								int N,
								int M,
								int* pY,
								int* pC,
								float* pLam,
								bool* pA,
								float* pW,
								float* pGS,
								int* pColPtrs,
								int* pRowIndices,
								float* pLogQRatio
								)
{
	int x = threadIdx.x;
	int n = x + blockIdx.x*blockDim.x;
	
	__shared__ float logQratio[B];
	logQratio[x] = 0.0;
	
	if (n < N)
	{
		// the ratio is equal to log(num)-log(den), where each term consists
		// of a sum over spike children
		float num = 0;
		float den = 0;
		
		int cn = pC[n];
		int yn_old = pY[cn];
		int yn_new = (cn==k) ? m : yn_old;
		
		num += pLam[yn_new*N+n];
		den += pLam[yn_old*N+n];
		
		for (int pa_off=pColPtrs[n]; pa_off<pColPtrs[n+1]; pa_off++)
		{
			int n_pa = pRowIndices[pa_off];
			int c_pa = pC[n_pa];
			int y_pa_old = pY[c_pa];
			int y_pa_new = (c_pa==k) ? m : y_pa_old; 
			
			num += pA[y_pa_new*M+yn_new] * pW[y_pa_new*M+yn_new] * pGS[pa_off];
			den += pA[y_pa_old*M+yn_old] * pW[y_pa_old*M+yn_old] * pGS[pa_off];
		}
		
		// Update this spikes contribution to log Q-ratio
		logQratio[x] = logf(num)-logf(den);
	}
	
	// Sum over all the spikes in this block
	float logQratioSum = 0.0;
	reduceBlock(logQratio, &logQratioSum, OP_SUM);
	
	if (x==0)
	{
		pLogQRatio[blockIdx.x] = logQratioSum;
	}
}