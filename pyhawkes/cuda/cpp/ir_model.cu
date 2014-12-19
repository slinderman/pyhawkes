#include <cuda.h>
#include <math_constants.h>

#define OP_SUM  0
#define OP_MULT 1

#define G_LOGISTIC_NORMAL 0

#define LOGISTIC_NORMAL_SUM         0
#define LOGISTIC_NORMAL_VARIANCE    1
#define DEBUG_SUM_DS                2

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
 * Sum the number of nonzero entries in each block of Z
 */
__global__ void sumNnzZPerBlock(int K,
                                int N,
                                int* pZ,
                                int* pC,
                                int* nnzZ)
{
    int ki    = blockIdx.x;     // W,A matrix row
    int kj    = blockIdx.y;     // W,A matrix column
    int k_ind = ki*K + kj;      // linear index into W,A
    int j     = threadIdx.x;    // Z_{ki,kj} column, or columns if
                                // Ns[j] is greater than blocksize


    // Otherwise we must sum the nonzero entries of Z and reduce
    // First calculate the number of nonzero entries per thread.
    // and store in shared memory
    __shared__ float nnz[B];
    nnz[j] = 0.0;
            
    for (int jj=j; jj<N; jj+=B)
    {   
        if (pZ[jj]>-1 && pC[jj]==kj && pC[pZ[jj]]==ki)
        {
            nnz[j] += 1.0;
        }
    }

    // Reduce the nnz matrix
    float nnzSum;
    reduceBlock(nnz, &nnzSum, OP_SUM);

    // First thread sets the output
    if (j==0)
    {
        nnzZ[k_ind] = (int)nnzSum;
    }
}

/*
 * Compute sufficient statistics to update the parameters of the
 * impulse response kernel G
 */
__global__ void computeGSuffStatistics(int K,
		                               int N, 
		                               int gType,
                                       int gStatistic,
                                       float* pgStatisticParam,
                                       float dt_max,
                                       int* pC,
                                       float* pDS,
                                       int* pRows,
                                       int* pCols,
                                       int* pZ,
                                       bool* pA,
                                       float* pStatistic)
{
    int ki    = blockIdx.x;     // A matrix row
    int kj    = blockIdx.y;     // A matrix column
    int k_ind = ki*K + kj;      // linear index into A
    int j     = threadIdx.x;    // Z_{ki,kj} column, or columns if
                                // Ns[j] is greater than blocksize

    // If there is no edge at this block then there
    // cannot be parents attributed across the pairs
    if (!pA[k_ind])
    {
        if (j==0)
        {
            pStatistic[k_ind] = 0.0;
        }
        return;
    }
    
    __shared__ float statistic[B];

    statistic[j] = 0.0;

    for (int jj=j; jj<N; jj+=B)
    {
        // for each spike identify its parent and increment dS sum
        // with the corresponding delta spike time

    	int par = pZ[jj];
        if (par>-1 && pC[jj]==kj && pC[par]==ki)
        {
            // Find the parent in column jj
            for (int ii=pCols[jj]; ii<pCols[jj+1]; ii++)
            {
                bool parFound = false;
                
                if (pRows[ii]==par)
                {
                    parFound = true;
                    
                    float x = 0.0;
                    float x_bar = 0.0;
                    switch(gType)
                    {
                    case G_LOGISTIC_NORMAL:
                        x = logf(pDS[ii]) - logf(dt_max-pDS[ii]);
                        switch(gStatistic)
                        {
                        case LOGISTIC_NORMAL_SUM:
                            statistic[j] += x;
                            break;
                        case LOGISTIC_NORMAL_VARIANCE:
                            x_bar = pgStatisticParam[k_ind];
                            statistic[j] += powf(x-x_bar, 2.0);
                            break;
                        case DEBUG_SUM_DS:
                            statistic[j] += 1;
                            break;
                        default:
                            //TODO: throw exception
                            break;
                        }
                            
                        break;
                    default:
                        //TODO: throw exception
                        break;
                    }
                    
                    if(parFound)
                    {
                        break;
                    }
                }
            }
        }
    }

    float blockStatistic;
    reduceBlock(statistic, &blockStatistic, OP_SUM);

    // First thread sets the output
    if (j==0)
    {
        pStatistic[k_ind] = blockStatistic;
    }
}

/*
 * Apply g(x; mu,tau) to every entry in dS, where x
 * is a normally distributed random variable according to
 * mu and precision tau. The density g is passed through a
 * logistic transformation. Run this kernel with
 * a single thread for every entry in pDS. 
 */
__global__ void computeLogisticNormalGS(float* pDS,
                                        int dS_len,
                                        float mu,
                                        float tau,
                                        float dt_max,
                                        float* pGS)
{
    int j = blockIdx.y * gridDim.x * blockDim.x +
            blockIdx.x * blockDim.x +
            threadIdx.x;
    
    if (j >= dS_len)
    {
        return;
    }
    
    float ds = pDS[j];
    float normpdf = sqrtf(tau/(2*CUDART_PI_F))*
                    expf(-1.0*tau/2.0*powf(logf(ds)-logf(dt_max-ds)-mu, 2.0));

    // Handle boundary conditions where g(t) is undefined
    // At those points g approaches 0.0 in the limit
    if (ds<=0.0|| ds>=dt_max)
    {
        pGS[j] = 0.0;
    }
    else
    {
        pGS[j] = dt_max/(ds*(dt_max-ds))*normpdf;
    }
}

/*
 * Apply g(x; mu,tau) to every entry in dS, where x
 * is a normally distributed random variable according to
 * mu and precision tau. The density g is passed through a
 * logistic transformation. Run this kernel with
 * a single thread for every entry in pDS. 
 */
__global__ void computeLogisticNormalGSIndiv(int K,
		                                     int N,
											 int* pC,
											 int* pRows,
											 int* pCols,
											 float* pMu,
											 float* pTau,
											 float dt_max,
											 float* pDS,
											 float* pGS)
{    
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= N)
    {
        return;
    }
    
	// For each ISI, determine the parent and child processes
    int cj = pC[j];
	for (int i=pCols[j]; i<pCols[j+1]; i++)
	{	
		int ci = pC[pRows[i]];
		
		float mu = pMu[ci*K+cj];
		float tau = pTau[ci*K+cj];
		
		float ds = pDS[i];
		float normpdf = sqrtf(tau/(2*CUDART_PI_F))*
						expf(-1.0*tau/2.0*powf(logf(ds)-logf(dt_max-ds)-mu, 2.0));

		// Handle boundary conditions where g(t) is undefined
		// At those points g approaches 0.0 in the limit
		if (ds<=0.0|| ds>=dt_max)
		{
			pGS[i] = 0.0;
		}
		else
		{
			pGS[i] = dt_max/(ds*(dt_max-ds))*normpdf;
		}
	}
}