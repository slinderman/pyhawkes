#include <cuda.h>

#define OP_SUM  0
#define OP_MULT 1

#define ERROR_SUCCESS               0
#define ERROR_MAX_HIST_INSUFFICIENT 1
#define ERROR_INVALID_PARAMETER     2
#define ERROR_SAMPLE_FAILURE        3

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

/*
 * Sample a Gamma RV using the Marsaglia Tsang algorithm. This
 * is much faster than algorithms based on transcendental functions used
 * by Numpy. We do have some overhead from generating extra unif
 * and normal RVs that are just rejected.
 * Our assumption is that W.H.P. we will successfully generate
 * a RV on at least one of the 1024 threads per block.
 * 
 * The vanilla Marsaglia alg requires alpha > 1.0
 * pU is a pointer to an array of uniform random variates, 
 * one for each thread. pN similarly points to normal
 * random variates.
 */
__global__ void sampleGammaRV(float* pU,
                              float* pN,
                              float* pAlpha,
                              float* pBeta,
                              float* pG,
                              int* pStatus)
{
    int x = threadIdx.x;
    int ki = blockIdx.x;
    int kj = blockIdx.y;
    int k_ind = ki*gridDim.y + kj;
    float u = pU[k_ind*blockDim.x + x];
    float n = pN[k_ind*blockDim.x + x];
    
    __shared__ float gamma[B];
    __shared__ bool accept[B];
    
    accept[x] = false;
    
    float a = pAlpha[k_ind];
    float b = pBeta[k_ind];
    
    if (a < 1.0)
    {
        if (x==0)
        {
            pStatus[k_ind] = ERROR_INVALID_PARAMETER;
        }
        return;
    }
    
    float d = a-1.0/3.0;
    float c = 1.0/sqrtf(9.0*d);
    float v = powf(1+c*n,3);

    // if v <= 0 this result is invalid
    if (v<=0)
    {
        accept[x] = false;
    }
    else if (u <=(1-0.0331*powf(n,4.0)) ||
             (logf(u)<0.5*powf(n,2.0)+d*(1-v+logf(v))))
    {
        // rejection sample. The second operation should be
        // performed with low probability. This is the "squeeze"
        gamma[x] = d*v;
        accept[x] = true;
    }
    
    // Reduce across block to find the first accepted sample
    __syncthreads();
    int half = blockDim.x / 2;
    if( x < half )
    {
       while( half > 0 )
       {
           if(x < half)
           {
               // if the latter variate was accepted but the current
               // was not, copy the latter to the current. If the current
               // was accepted we keep it. If neither was accepted we
               // don't change anything.
               if (!accept[x] && accept[x+half])
               {
                   gamma[x] = gamma[x+half];
                   accept[x] = true;
               }
           }
           half = half / 2;
           __syncthreads();
       }
    }
    
    // Store the sample to global memory, or return error if failure
    if (x == 0)
    {
        if (accept[0])
        {
            // rescale the result (assume rate characterization)
            pG[k_ind] = gamma[0]/b;
            pStatus[k_ind] = ERROR_SUCCESS;
        }
        else
        {
            pStatus[k_ind] = ERROR_SAMPLE_FAILURE;
        }
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

/**
 * Compute the posterior distribution on W_{ki,kj} where ki and
 * kj are the process IDs, and are mapped uniquely to blocks.
 * Thread index represents columns, j, of the submatrix of Z.
 * The first thread is responsible for writing a value to the
 * output matrices pAlpha_w_post, pBeta_w_post.
 */
__global__ void computeWPosterior(int K,
		                          int* pNnz,
                                  int* pNs,
                                  bool* pA, 
                                  float alpha_w0, 
                                  float beta_w0,
                                  float* pAlpha_w_post,
                                  float* pBeta_w_post
                                  )
{
    
    int ki    = blockIdx.x*blockDim.x + threadIdx.x;
    int kj    = blockIdx.y*blockDim.y + threadIdx.y;
    int k_ind = ki*K + kj;
    
    if (ki >= K || kj >= K)
    {
        return;
    }
    else if (!pA[k_ind])
    {
        // If A[ki,kj]==0 then we don't need to calculate anything
        pAlpha_w_post[k_ind] = alpha_w0;
        pBeta_w_post[k_ind]  = beta_w0;     
        return;
    }
    else
    {
        pAlpha_w_post[k_ind] = alpha_w0 + (float)pNnz[k_ind];
        pBeta_w_post[k_ind]  = beta_w0 + (float)(pNs[ki]);
    }
}

/**
 * Compute the posterior params for a symmetric prior on W.
 * Due to conjugacy, the params are just the sum of the two symmetric
 * etnries
 */
__global__ void computeSymmWPosterior(int K,
                                      float alpha_w0, 
                                      float beta_w0,
                                      float* pAlpha_w_post,
                                      float* pBeta_w_post
									  )
{
    
    int ki    = blockIdx.x*blockDim.x + threadIdx.x;
    int kj    = blockIdx.y*blockDim.y + threadIdx.y;
    
    // Only work on the upper triangular entries
    if (ki > K || kj > K || kj <= ki)
    {
        return;
    }
    else
    {
    	int upper_ind = ki*K+kj;
    	int lower_ind = kj*K+ki;
    	
    	// Do not double count the prior
        float alpha_post = pAlpha_w_post[upper_ind] + pAlpha_w_post[lower_ind] - alpha_w0;
        float beta_post = pBeta_w_post[upper_ind] + pBeta_w_post[lower_ind] - beta_w0;
        
        // Update the entries
        pAlpha_w_post[upper_ind] = alpha_post;
        pAlpha_w_post[lower_ind] = alpha_post;
        pBeta_w_post[upper_ind]  = beta_post;
        pBeta_w_post[lower_ind]  = beta_post;
    }
}

/** 
 * Copy the newly sampled W's across diagonal to make them symmetric
 */
__global__ void copySymmW(int K,
		                  float* pW
		                  )
{
	int ki    = blockIdx.x*blockDim.x + threadIdx.x;
	int kj    = blockIdx.y*blockDim.y + threadIdx.y;
	
	// Only work on the upper triangular entries
	if (ki > K || kj > K || kj <= ki)
	{
		return;
	}
	else
	{
		int upper_ind = ki*K+kj;
		int lower_ind = kj*K+ki;
		
		pW[lower_ind] = pW[upper_ind];
	}
}