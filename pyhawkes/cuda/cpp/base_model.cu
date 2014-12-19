#include <cuda.h>
#include <math_constants.h>

#define OP_SUM  0
#define OP_MULT 1

#define MH_ADD  0
#define MH_DEL  1
#define MH_NOP  2

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
 * Helper function to sum across a block.
 * Assume pS_data is already in shared memory
 * Only the first thread returns a value in pSum
 */
__device__ void reduceBlockDouble( double pSdata[B], double* pSum, int op )
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
 * Compute the weighted sum along columns of G as required for
 * sampling new parents. 
 * This should be launched on a KxN/B grid of 1024x1 blocks.
 */
__global__ void computeWGSForAllSpikes(int K,
		                               int N,
                                       int* pC,
                                       float* pGS,
                                       int* pCols,
                                       int* pRowInds,
                                       float* pW,
                                       bool* pA,
                                       float* pWGS
                                       )
{
    int ki = blockIdx.y;
    int j  = blockIdx.x*blockDim.x + threadIdx.x;
        
    if (j<N)
    {
        int kj = pC[j];
        int k_ind = ki*K+kj;
        
        // W is always populated, regardless of whether or not an edge exists
//        float w = pA[k_ind] ? pW[k_ind] : 0.0;
        float w = pW[k_ind];
        
        // If the effective weight is greater than 0, compute the weighted
        // sum along the column of GS
//        if (w > 0.0)
//        {
            float gs_sum = 0.0;
            // Iterate over row entries for dS column jj
            // Since dS is stored in CSC format, pCols gives the
            // pointers into dS for each column
            for (int gs_off=pCols[j]; gs_off<pCols[j+1]; gs_off++)
            {
            	if (pC[pRowInds[gs_off]]==ki)
            	{
            		gs_sum += pGS[gs_off];
            	}
            }
            
            // Multiply by the current weight
            pWGS[ki*N+j] = w * gs_sum;
//        }
//        else
//        {
//            pWGS[ki*N+j] = 0.0;
//        }
    }
}

/**
 * Compute the weighted sum along columns of G for a single block.
 * This overrides the adjacency matrix and assumes an edge exists.
 * This is necessary for computation of the Q ratio when a new edge
 * is proposed.
 * This should be launched on a 1xN/B grid of 1024x1 blocks.
 */
__global__ void computeWGSForNewEdge(int ki,
                                     int kj,
                                     int K,
                                     int N,
                                     int* pC,
                                     float* pGS,
                                     int* pCols,
                                     int* pRowInds,
                                     float* pW,
                                     float* pWGS
                                     )
{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int k_ind = ki*K + kj;
    int wgsOffset = ki*N;
    
    float w = pW[k_ind];
    
    if (j<N && pC[j]==kj)
    {
        float gs_sum = 0.0;
        // Iterate over row entries for dS column jj
        // Since dS is stored in CSC format, pCols gives the
        // pointers into dS for each column
        for (int gs_off=pCols[j]; gs_off<pCols[j+1]; gs_off++)
        {
        	if (pC[pRowInds[gs_off]]==ki)
        	{
        		gs_sum += pGS[gs_off];
        	}
        }
        
        // Multiply by the current weight
        pWGS[wgsOffset+j] = w * gs_sum;
    }
}

/**
 * Clear the WGS entries for a deleted edge
 * This should be launched on a 1xNs[kj]/B grid of 1024x1 blocks.
 */
__global__ void clearWGSForDeletedEdge(int ki,
                                       int kj,
                                       int N,
                                       int* pC,
                                       float* pWGS
                                       )
{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int wgsOffset = ki*N;

    if (j<N && pC[j]==kj)
    {
        pWGS[wgsOffset+j] = 0.0;
    }
}

/**
 * Compute the MH acceptance probability of adding edge (i,j)
 * This should be launched on a grid of (1xceil(Ns[kj]/B)) blocks
 * each with 1024 threads. The output should be a float vector
 * of size (1x ceil(Ns[kj]/B))
 * 
 * For stability we compute this in log space
 * 
 */
__global__ void computeProdQratio(int kj,
		                          int K,
                                  int N,
                                  int* pC,
                                  float* pWGS,
                                  float* pLam,
                                  int affectedRow,
                                  int mhOp,
                                  double* qr
                                  )
{     
    int x = threadIdx.x;
    int j = blockIdx.x * blockDim.x + x;
        
    // The ratio of Q's invovles a product over all spikes
    // we use shared memory to store the partial products each 
    // thread computes and then reduce as above
    __shared__ double logQrat[B];
    logQrat[x] = 0.0;
    
    if (j<N && pC[j]==kj)
    {
        float lam = pLam[kj*N+j];
        double wgsNum = lam;        // numerator: sum over WGS's current and changed rows
        double wgsDen = lam;        // denomenator: sum over WGS's current rows
        for (int ki=0; ki<K; ki++)
        {
            int wgsOffset = ki*N;
            
            if (ki != affectedRow)
            {
                // both numerator and denominator match
                wgsNum += pWGS[wgsOffset+j];
                wgsDen += pWGS[wgsOffset+j];
            }
            else if (mhOp == MH_ADD)
            {
                // WGS computed with proposed edge weight
                // Current rows sum to 0 since no edge present before
                wgsNum += pWGS[wgsOffset+j];
            }
            else
            {
                // Edge is being removed
                // WGS was computed with current edge and 
                // weight does not contribute to numerator
                wgsDen += pWGS[wgsOffset+j];
            }
        }
        
        logQrat[x] = logf(wgsNum) - logf(wgsDen);
    }
    
    // Take the product of Qrat by reduction
    double sumLogQrat = 0.0;
    reduceBlockDouble(logQrat, &sumLogQrat, OP_SUM);
    
    if (x==0)
    {
        qr[blockIdx.x] = sumLogQrat;
    }
}

__global__ void computeLkhdRatioA(int ki,
					              int kj,
		                          int K,
                                  int N,
                                  bool* pA,
                                  int* pC,
                                  float* pWGS,
                                  float* pLam,
                                  float* lr
                                  )
{     
    int x = threadIdx.x;
    int j = blockIdx.x * blockDim.x + x;
        
    // The ratio of Q's invovles a product over all spikes
    // we use shared memory to store the partial products each 
    // thread computes and then reduce as above
    __shared__ float logLkhdRat[B];
    logLkhdRat[x] = 0.0;
    
    if (j<N && pC[j]==kj)
    {
        float lam = pLam[kj*N+j];
        float wgsNum = lam;        // numerator: sum over WGS's current and changed rows
        float wgsDen = lam;        // denomenator: sum over WGS's current rows
        for (int ii=0; ii<K; ii++)
        {
            int wgsOffset = ii*N;
            
            if (ii != ki)
            {
            	if (pA[ii*K+kj])
            	{
					// both numerator and denominator match
					wgsNum += pWGS[wgsOffset+j];
					wgsDen += pWGS[wgsOffset+j];
            	}
            }
            else
            {
            	// Only add WGS to the numerator for potential edge ki
            	wgsNum += pWGS[wgsOffset+j];
            }
        }
        
        logLkhdRat[x] = logf(wgsNum) - logf(wgsDen);
    }
    
    // Take the product of Qrat by reduction
    float sumLogLkhdRat = 0.0;
    reduceBlock(logLkhdRat, &sumLogLkhdRat, OP_SUM);
    
    if (x==0)
    {
        atomicAdd(lr,sumLogLkhdRat);
    }
}

/**
 * Sample a random value of A[ki,kj] given the two terms
 * which add up to the log odds ratio and a random variate p
 */
__global__ void sampleA(int ki,
		                int kj,
		                int K,
		                bool* pA,
		                float logpratio,
		                float* pLogqratio,
		                float p
		                )
{
	float logit_pr_A = logpratio + (*pLogqratio);
	
	// compute log_pr_A using log-sum-exp trick
	// log_pr_A = -logf(1.0+expf(-logit_pr_A))
	float log_pr_A = 0.0;
	if (logit_pr_A < 0.0)
	{
		log_pr_A = logit_pr_A - logf(1.0+expf(logit_pr_A));
	}
	else
	{
		log_pr_A = -logf(1.0+exp(-logit_pr_A));
	}
	
	// Sample A
	if (logf(p) < log_pr_A)
	{
		pA[ki*K+kj] = true;
	}
	else
	{
		pA[ki*K+kj] = false;
	}
}

/**
 * Assign new parents neurons for each spike on k
 * Launch this on 1xceil(N/B) grid of blocks
 */
__global__ void sampleNewParentProcs(int K,
		                             int N,
                                     int* pC,
                                     bool* pA,
                                     float* pW,
                                     float* pWGS,
                                     float* pLam,
                                     float* urand,
                                     int* pZ_temp
                                     )
{
    // TODO: index using both x and y for large datasets
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    
    // Sum columns of WGS, taking into account the deleted row
    // in the case of an edge removal
    if (j<N)
    {
    	int kj = pC[j];
        float lam = pLam[kj*N+j];
    
        // Clear whatever existing parent might exist
        pZ_temp[j] = -1;
        
        // In the first pass compute the cumulative sum.
        // In the second pass we'll use this information to
        // determine which process contains the parent spike.
        float wgsSum = lam;
        for (int ki=0; ki<K; ki++)
        {
        	if (pA[ki*K+kj])
        	{
        		wgsSum += pWGS[ki*N+j];
        	}
        }

        // Check if this block is responsible for assigning
        // a new parent based on the value of urand
        float wu = urand[j] * wgsSum;
        if (wu <= lam)
        {
            pZ_temp[j] = -1;
            return;
        }

        wgsSum = lam;
        for (int ki=0; ki<K; ki++)
        {
        	if (pA[ki*K+kj])
			{
        		wgsSum += pWGS[ki*N+j];
			}
            if (wgsSum >= wu)
            {
                pZ_temp[j] = ki;
                break;
            }
        }
    }
}

/**
 * Assign new parents Z for each spike on a given process.
 * This should be called with a grid of K x ceil(N/B)) blocks,
 * each of which has B threads.
 * Assume that an array of Ns[kj] uniform random
 * numbers between [0,1] is present in the array urand. These will
 * be used to identify the parent neuron and then the parent spike.
 */
__global__ void sampleNewParentSpikes(int K,
		                              int N,
                                      int* pC,
                                      int* pZ_temp,
                                      float* pGS,
                                      int* pRowIndices,
                                      int* pCols,
                                      float* pW,
                                      float* pWGS,
                                      int* pZ,
                                      float* urand
                                      )
{
    int j     = blockIdx.x*blockDim.x + threadIdx.x;
    int ki    = blockIdx.y;
    
    
    // Sum columns of WGS, taking into account the deleted row
    // in the case of an edge removal
    if (j<N)
    {
    	// Find kj 
        int kj=pC[j];
        int k_ind = ki*K + kj;
        // Clear whatever existing parent might exist
        if (ki==0 && pZ_temp[j]==-1)
        {
            pZ[j] = -1;
            return;
        }

        if (pZ_temp[j]==ki)
        {
            // Check if this block is responsible for assigning
            // a new parent based on the value of urand         
            // Update u so that in the subsequent step we can
            // simply iterate over spikes
            float gu = urand[j] * pWGS[ki*N+j]/pW[k_ind];

            // Iterate over spikes on ki to find a new parent
            // We can run into trouble if gu ~= 1.0. To avoid this,
            // only iterate to the second to last spike.
            float cumPr = 0.0;
            for (int GS_off=pCols[j]; GS_off < pCols[j+1]; GS_off++)
            {
            	if (pC[pRowIndices[GS_off]]==ki)
            	{
					cumPr += pGS[GS_off];
					// If the cumulative GS sum exceeds the threshold, this is the parent
					if (cumPr > gu)
					{
						pZ[j] = pRowIndices[GS_off];
						break;
					}
            	}
            }
            
        }
    }
}

/**
 * Compute the log likelihood term corresponding to each spike. 
 * Namely, log of the instantaneous intensity of the process which
 * parented the spike.
 */
__global__ void computeLogLkhdPerSpike(int K,
		                               int N,
                                       int* pC,
                                       float* pLam,
                                       bool* pA,
                                       float* pW,
                                       float* pGS,
                                       int* pCols,
                                       int* pRowIndices,
                                       float* pLL
                                       )
{
    int x = threadIdx.x;
    int n = blockIdx.y * gridDim.x * blockDim.x +
            blockIdx.x * blockDim.x +
            threadIdx.x;
    
    __shared__ float ll[B];
    ll[x] = 0.0;
    
    if (n<N)
    {       
        // TODO: Use computeWGSForAllSpikes 
        // Log likelihood is the log of the total rate
        int c_n = pC[n];
        float lambda = pLam[c_n*N+n];
        for (int row=pCols[n]; row<pCols[n+1]; row++)
        {
            int row_ind = pRowIndices[row];
            int c_row_ind = pC[row_ind];
            int c_ind = c_row_ind*K + c_n;
            lambda += (pA[c_ind] * pW[c_ind] * pGS[row]);
        }
        
        ll[x] = logf(lambda);
    }
    else
    {
        ll[x] = 0.0;
    }
    
    // Sum up the log likelihoods
    float llSum;
    reduceBlock(ll, &llSum, OP_SUM);
    
    // First thread in the block sets the output
    if (x==0)
    {
        int block = blockIdx.y * gridDim.x + blockIdx.x;
        pLL[block] = llSum;
    }
    
}


/**
 * Compute the log likelihood term corresponding to each spike. 
 * Namely, log of the instantaneous intensity of the process which
 * parented the spike.
 */
__global__ void computeConditionalIntensity(int K,
                                            int Nt,
                                            int* pC,
                                            bool* pA,
                                            float* pW,
                                            float* pGS,
                                            int* pCols,
                                            int* pRowIndices,
                                            float* pCond
                                            )
{
    int t = threadIdx.x + blockIdx.x*blockDim.x;
    int c_t = blockIdx.y;
    
    if (t < Nt)
    {
    	pCond[c_t*Nt+t] = 0;
        // Iterate over potential spike parents
        for (int row=pCols[t]; row<pCols[t+1]; row++)
        {
            int n = pRowIndices[row];
            int c_n = pC[n];
            if (pA[c_n*K+c_t])
            {
                pCond[c_t*Nt+t] += pW[c_n*K+c_t] * pGS[row];
            }
        }
    }
}
