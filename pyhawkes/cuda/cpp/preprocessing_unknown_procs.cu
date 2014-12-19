#include <cuda.h>

#define G_LOGISTIC_NORMAL 0

#define ERROR_SUCCESS               0
#define ERROR_MAX_HIST_INSUFFICIENT 1
#define ERROR_INVALID_PARAMETER     2
#define ERROR_SAMPLE_FAILURE        3


const int B = %(B)s;               // block size

/**
 * Helper function to binary search through an array.
 * If lowerExclusive is true, we return ind such that
 * pArr[ind-1] <= target < pArr[ind]. 
 * Otherwise we return the upper exclusive index s.t.
 * pArr[ind] < target <= pArr[ind+1]  
 */
__device__ void binarySearch(float pTarget[B], 
                             float* pArr,
                             int offset,
                             int N,
                             int lb_init,
                             int ub_init,
                             bool lowerExclusive,
                             int pInd[B]
                             )
{
    int j = threadIdx.x;
    
    int lb = lb_init;
    int ub = ub_init;
    
//    // check if this thread is valid
//    if (pTarget[j] < 0)
//    {
//        pInd[j] = -1;
//        return;
//    }
//    
    // Check boundary conditions to ensure initial invariances
    if (lowerExclusive)
    {
    	// lowerExclusive must satisfy pArr[ind-1] <= target < pArr[ind] 
		if (pTarget[j] < pArr[offset+lb])
		{
			pInd[j] = lb;
			return;
		}
		
		if (pTarget[j] >= pArr[offset+ub])
		{
			// no index exists satisfying  pArr[ind-1] <= target < pArr[ind] 
			pInd[j] = ub+1;
			return;
		}
    }
    else
    {
    	// upperExclusive must satisfy pArr[ind] < target <= pArr[ind+1] 
    	if (pTarget[j] <= pArr[offset+lb])
		{
			// no index exists satisfying pArr[ind] < target <= pArr[ind+1]  
			pInd[j] = lb-1;
			return;
		}
		
		if (pTarget[j] > pArr[offset+ub])
		{
			pInd[j] = ub;
			return;
		}
    }
    
    // start at midpoint and move to the left or right depending on whether 
    // the target is above or below the current index. 
    // If lowerExclusive the bounds are such that lb <= target, ub > target
    // If not lowerExclusive (upperExclusive) the bounds are such that lb < target, ub >= target
    int max_rounds = N;
    int round = 0;
    while (ub-lb > 1)
    {
    	round++;
    	if (round >= max_rounds)
    	{
    		break;
    	}
    	
        int i = (int)((ub+lb)/2);
		if (pArr[offset+i] > pTarget[j])
		{
			ub = i;
			continue;
		}
		else if(pArr[offset+i] < pTarget[j])
		{
			lb = i;
			continue;
		}
		else
		{
			// pArr[offset+i] == pTarget[j]
			// maintain the appropriate invariance condition
			if (lowerExclusive)
			{
				lb = i;
			}
			else
			{
				ub = i;
			}
		}

    }
        
    // Return the correct index. The variable names are somewhat misleading, lowerExclusive
    // means return the index such that pArr[ind] is the first index greater than the target.
    // since lb <= tgt, ub > tgt, and ub-lb <= 1, either target equals lb or target is between lb and ub
    // therefore we return ub. If not lowerExclusive, do the opposite
    if (lowerExclusive)
    {
    	pInd[j] = ub;
    }
    else
    {
    	pInd[j] = lb;
    }
}


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
                                   int N1,
                                   float* pS1,
                                   int N2, 
                                   float* pS2, 
                                   int* pColStart,
                                   int* pColEnd,
                                   int* pColSizes
                                   )
{
    int x = threadIdx.x;
    int j = blockIdx.x*blockDim.x + x;

    __shared__ float pColStartTgt[B];
    __shared__ int pColStartInd[B];
    __shared__ float pColEndTgt[B];
    __shared__ int pColEndInd[B];
    
    
    // Binary search to find the start and end of the column
    if (j<N2)
    {
        pColStartTgt[x] = pS2[j] - dt_max;
        pColEndTgt[x] = pS2[j];
    }
    else
    {
        pColStartTgt[x] = -1;
        pColEndTgt[x] = -1;
    }
    binarySearch(pColStartTgt, pS1, 0, N1, 0, N1-1, true,  pColStartInd);
    binarySearch(pColEndTgt, pS1, 0, N1, pColStartInd[x], N1-1, false, pColEndInd);
     
    
    // The results of the binary search are inclusive indices (assuming both are nonnegative)
    if (j<N2)
    {    	
    	if (pColStartInd[x] <= pColEndInd[x])
    	{
    		
    		pColSizes[j] = pColEndInd[x] - pColStartInd[x] + 1;
    	}
    	else
    	{
    		pColSizes[j] = 0;
    	}
    	
    	pColStart[j] = pColStartInd[x];
		pColEnd[j] = pColEndInd[x];
    }
}



/**
 * Compute the row indices and dS values for each block's sparse matrix.
 * pIndices[j]=i if j is the i'th spike on process pC[j].
 */
__global__ void computeRowIndicesAndDs(int gType,
                                       float* pS1,
                                       int N2, 
                                       float* pS2,
                                       int* pColStart,
                                       int* pColEnd,
                                       int* pColPtrs,
                                       int* pRowIndices,
                                       float* pDs
                                       )
{
    int x  = threadIdx.x;
    int j  = blockIdx.x*blockDim.x + x;
    
    
    // We should now have the spike history in shared memory
    // each thread iterates back in shared memory, updating its
    // spike count per neuron as it goes
    if (j<N2)
    {
        // iterate backwards, starting at the previous spike
        int row_ind = 0;
        for (int row=pColStart[j]; row<=pColEnd[j]; row++)
        {
			pRowIndices[pColPtrs[j]+row_ind] = row;
			
			// TODO: Store different statistics depending on the type of kernel in use
			//       Note that this could exceed our memory capacity
			switch(gType)
			{
			case G_LOGISTIC_NORMAL:
				pDs[pColPtrs[j]+row_ind] = pS2[j]-pS1[row];
				break;
			default:
				// TODO: Throw exception
				break;
			}
			row_ind++;
        }
    }
}

/**
 * Compute the spatial offsets for each pair of spikes
 */
__global__ void computeDx(int D,
		                  int N1,
						  float* pX1,                               
						  int N2, 
						  float* pX2,
						  int* pRowIndices,
						  int* pColPtrs,
						  float* pDx
						  )
{   
	int x = threadIdx.x;
	int j = blockIdx.x*blockDim.x + x;

    if (j<N2)
    {
        // this thread is associated with a single spike in dataset2 and all spikes 
		// in dataset 1 on process ki. Iterate over each spike parent in dataset 1.
        for (int pair_ind=pColPtrs[j]; pair_ind<pColPtrs[j+1]; pair_ind++)
        {
        	// get the parent spike in [0,N1)
        	int pa = pRowIndices[pair_ind];
        	
        	// Iterate over each dimension of the spatial data
        	// recall pX2 and pX1 are DxN matrices in row-major order
			for (int d=0; d<D; d++)
			{
				pDx[D*pair_ind+d] = pX2[d*N2+j] - pX1[d*N1+pa];
			}
        }
    }
}