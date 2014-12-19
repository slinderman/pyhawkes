/**
 * Unfortunately these have to be copied into the individual source files
 * since pycuda doesn't seem to handle includes very well.
 */

extern const int B;
extern const int K;

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
 * is much faster than algorithms based on Mersenne twister used
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
