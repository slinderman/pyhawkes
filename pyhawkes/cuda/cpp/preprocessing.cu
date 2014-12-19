#include <cuda.h>

#define G_LOGISTIC_NORMAL 0

#define ERROR_SUCCESS               0
#define ERROR_MAX_HIST_INSUFFICIENT 1
#define ERROR_INVALID_PARAMETER     2
#define ERROR_SAMPLE_FAILURE        3

// constants to be filled in before compiling
// number of processes
const int K = %(K)s;
// max number of spikes in history
const int MAX_HIST = %(MAX_HIST)s;

/**
 * Compute the amount of memory necessary for each process pair.
 * pHistSz must be a KxN array. Each thread is responsible for
 * populating a column of this output matrix.
 * 
 * This function populates the KxN pColSizes matrix with the number
 * of spikes on process ki which affect spike nj. We assume that 
 * all such spikes come after nj-MAX_HIST. 
 * 
 * pIndices[j]=i if j is the i'th spike on process pC[j].
 * 
 * If MAX_HIST is insufficient return an error status
 */
__global__ void computeColumnSizes(float dt_max,
                                   int N, 
                                   int* pCumSumNs,
                                   float* pS,                               
                                   int* pIndices,
                                   int* pC,
                                   int* pColSizesBuffer,
                                   int* pStatus
                                   )
{
    int x  = threadIdx.x;
    int ki = blockIdx.x;
    int j0 = blockIdx.y*blockDim.x;
    int j  = j0 + x;
    int kj = pC[j];
    int blockInd = blockIdx.x * gridDim.y + blockIdx.y;

    // Use shared memory to store the relevant spikes
//    __shared__ float S[B+MAX_HIST];
//    __shared__ int C[B+MAX_HIST];
    
    // If MAX_HIST is insufficient it will show on the first spike
    if (x==0)
    {
        pStatus[blockInd] = ERROR_SUCCESS;
        if (j > MAX_HIST)
        {
            if (pS[j-MAX_HIST-1] > (pS[j]-dt_max))
            {
                pStatus[blockInd] = ERROR_MAX_HIST_INSUFFICIENT;
                __syncthreads();
                return;
            }
        }
        __syncthreads();
    }
    else
    {
        // Check if the first thread failed. If so, exit. 
        __syncthreads();
        if (pStatus[blockInd] == ERROR_MAX_HIST_INSUFFICIENT)
        {
            return;
        }
    }
    
//    // Load spike history into memory
//    for (int offset=x; offset<B+MAX_HIST; offset+=B)
//    {
//        int i = j0-MAX_HIST+offset;
//        if (i>=0 && i<N)
//        {
//            S[offset] = pS[i];
//            C[offset] = pC[i];
//        }
//        else
//        {
//            S[offset] = -1.0;
//            C[offset] = -1;
//        }
//    }
//    __syncthreads();
//
    // We should now have the spike history in shared memory
    // each thread iterates back in shared memory, updating its
    // spike count per neuron as it goes
    if (j<N)
    {
        // count spikes that occurred on ki and influence j
        int col_size = 0;
        int buff_start = j0-MAX_HIST; 
        
        // iterate backwards, starting at the previous spike
        for (int buff_off=MAX_HIST+x-1; buff_off>=0; buff_off--)
        {
            // make sure this spike is valid
            // (these should be equivalent conditions)
//            if (buff_start + buff_off < 0 ||
//              buff_start + buff_off >= N ||
//                C[buff_off] == -1 ||
//                S[buff_off] < 0.0 ||
//                S[MAX_HIST+x]-S[buff_off] > dt_max)
            if (buff_start + buff_off < 0 ||
                buff_start + buff_off >= N)
            {
                // If this index is out of bounds
                break;
            }
            else if (pS[j]-pS[buff_start + buff_off] >= dt_max)
            {
                // If we've reached the end of the data window
                break;
            }
            else if (pS[j]-pS[buff_start + buff_off] == 0.0)
            {
                // If the 'previous' data point occur at exactly the same time
                // skip them. We only want dS > 0
                continue;
            }
            else if (pC[buff_start + buff_off]==ki)
            {
                col_size++;
            }
        }
        
        // Update global memory. We don't put it in the same order,
        // instead we index spikes according to which process they 
        // occurred on.
//        int* pColSizes = &pColSizesBuffer[ki*N+pCumSumNs[kj]];
//        pColSizes[pIndices[j]] = col_size;
        pColSizesBuffer[ki*N+pCumSumNs[kj]+pIndices[j]] = col_size;
    }
}

/**
 * Compute the col ptrs per block. These are simply the cumulative
 * sum of the col sizes. Use KxK grid of blocks, and one thread per 
 * block. We can compute the ColPtrOffsets entries a priori if we
 * know the cumulative sum of Ns. Each block (ki,kj) requires Ns[kj]+1
 * col ptrs, so there are N+K col_ptrs per row.
 * 
 * Launch this in a KxK grid with 1 thread per block.
 */
__global__ void computeColPtrs(int N,
                               int* pNs,
                               int* pCumSumNs,
                               int* pColSizesBuffer,
                               int* pColPtrsBuffer,
                               int* pColPtrOffsets
                               )
{
    int ki = blockIdx.x;
    int kj = blockIdx.y;
    int k_ind = ki*K+kj;
    
    int x = threadIdx.x;
    
    if (x==0)
    {
        // Set the colPtrOffsets for this block
        pColPtrOffsets[k_ind] = ki*(N+K) + pCumSumNs[kj] + kj;
        
        // Set the column pointers relative to the start of this
        // block's matrix
        int* pColSizes = &pColSizesBuffer[ki*N + pCumSumNs[kj]];
        int* pColPtrs = &pColPtrsBuffer[pColPtrOffsets[k_ind]];
        
        int cumSumColSizes = 0;
        for (int j=0; j<pNs[kj]; j++)
        {
            pColPtrs[j] = cumSumColSizes;
            cumSumColSizes += pColSizes[j];
        }
        
        // set the last col ptr
        pColPtrs[pNs[kj]] = cumSumColSizes;
    }
}

/**
 * Compute the size of the dS and rowIndices buffers that must
 * be allocated
 */
__global__ void computeDsBufferSize(int* pNs,
                                    int* pColPtrsBuffer,
                                    int* pColPtrOffsets,
                                    int* bufferSize)
{
    int x = threadIdx.x;
    int size = 0;
    
    if (x==0)
    {
        for (int ki=0; ki<K; ki++)
        {
            for (int kj=0; kj<K; kj++)
            {
                int k_ind = ki*K+kj;
                
                int* pColPtrs = &pColPtrsBuffer[pColPtrOffsets[k_ind]];
                // The last col ptr equals the number of data points
                size += pColPtrs[pNs[kj]];
            }
        }
        
        bufferSize[0] = size;
    }
}

/**
 * Compute the offsets into the row and data buffers using
 * the column pointers. We do this with a single block, single thread.
 */
__global__ void computeRowAndDsOffsets(int* pNs,
                                       int* pColPtrsBuffer,
                                       int* pColPtrOffsets,
                                       int* pRowAndDsOffsets)
{
    int offset = 0;
    
    for (int ki=0; ki<K; ki++)
    {
        for (int kj=0; kj<K; kj++)
        {
            int k_ind = ki*K+kj;
            pRowAndDsOffsets[k_ind] = offset;
            
            int* pColPtrs = &pColPtrsBuffer[pColPtrOffsets[k_ind]];
            // The last col ptr equals the number of data points
            offset += pColPtrs[pNs[kj]];
        }
    }
}

/**
 * Compute the row indices and dS values for each block's sparse matrix.
 * pIndices[j]=i if j is the i'th spike on process pC[j].
 */
__global__ void computeRowIndicesAndDs(float dt_max,
                                       int N, 
                                       int gType,
                                       float* pS,                               
                                       int* pIndices,
                                       int* pC,
                                       int* pColPtrsBuffer,
                                       int* pColPtrOffsets,
                                       int* pRowIndicesBuffer,
                                       float* pDsBuffer,
                                       int* pRowAndDsOffsets
                                       )
{
    int x  = threadIdx.x;
    int ki = blockIdx.x;
    int j0 = blockIdx.y*blockDim.x;
    int j  = j0 + x;
    int kj = pC[j];
    int k_ind = ki*K+kj;
    
    // Use shared memory to store the relevant spikes
//    __shared__ float S[B+MAX_HIST];
//    __shared__ float C[B+MAX_HIST];
   
//    // Load spike history into memory
//    for (int offset=x; offset<B+MAX_HIST; offset+=B)
//    {
//        int i = j0-MAX_HIST+offset;
//        if (i>=0 && i<N)
//        {
//            S[offset] = pS[i];
//            C[offset] = pC[i];
//        }
//        else
//        {
//            S[offset] = -1.0;
//            C[offset] = -1;
//        }
//    }
//    __syncthreads();
    
    // We should now have the spike history in shared memory
    // each thread iterates back in shared memory, updating its
    // spike count per neuron as it goes
    if (j<N)
    {
        // keep index ino row of the current column

        int* pRowIndices = &pRowIndicesBuffer[pRowAndDsOffsets[k_ind]];
        int* pColPtrs = &pColPtrsBuffer[pColPtrOffsets[k_ind]];
        float* pDs = &pDsBuffer[pRowAndDsOffsets[k_ind]];
        
        int col_ind = pIndices[j];
        int nnz_col = pColPtrs[col_ind+1]-pColPtrs[col_ind];
        int buff_start = j0-MAX_HIST; 
        
        
        // iterate backwards, starting at the previous spike
        int row_ind = 0;
        for (int buff_off=MAX_HIST+x-1; buff_off>=0; buff_off--)
        {
            if (buff_start + buff_off < 0 ||
                buff_start + buff_off >= N ||
                row_ind >= nnz_col)
            {
                // If this index is out of bounds or
                // we have reached the end of the column buffer
                break;
            }
            else if (pS[j]-pS[buff_start + buff_off] >= dt_max)
            {
                // If we've reached the end of the data window
                break;
            }
            else if (pS[j]-pS[buff_start + buff_off] == 0.0)
            {
                // If the 'previous' data point occur at exactly the same time
                // skip them. We only want dS > 0
                continue;
            }
            else if (pC[buff_start + buff_off]==ki)
            {
                pRowIndices[pColPtrs[col_ind]+nnz_col-1-row_ind] = pIndices[buff_start + buff_off];
                
                // TODO: Store different statistics depending on the type of kernel in use
                //       Note that this could exceed our memory capacity
                switch(gType)
                {
                case G_LOGISTIC_NORMAL:
                    pDs[pColPtrs[col_ind]+nnz_col-1-row_ind] = pS[j]-pS[buff_start+buff_off];
                    break;
                default:
                    // TODO: Throw exception
                    break;
                }
                row_ind++;
            }
        }
    }
}