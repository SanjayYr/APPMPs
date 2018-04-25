#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define TILE_SIZE 1024
// You can use any other block size you wish.
#define BLOCK_SIZE 512
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// Host Helper Functions (allocate your own data structure...)



// Device Functions



// Kernel Functions
__global__ void prescan(unsigned int *g_odata, unsigned int *g_idata, unsigned int *S, int n)
{
   __shared__ unsigned int temp[TILE_SIZE];  // allocated on invocation
   int tid = threadIdx.x;
   int bid = blockIdx.x;
   int index = bid * blockDim.x + tid;
   int offset = 1;

   temp[2*tid] = g_idata[2*index]; 
   temp[2*tid+1] = g_idata[2*index+1];

   for (int d = TILE_SIZE>>1; d > 0; d >>= 1)   
   { 
     __syncthreads();
     if (tid < d)
     {
        int ai = offset*(2*tid+1)-1;
        int bi = offset*(2*tid+2)-1;
        temp[bi] += temp[ai];
     }
     offset *= 2;
   }

   if (tid == 0)
   {
     if(S != NULL)
     {
        S[bid] = temp[TILE_SIZE - 1];
     }
     temp[TILE_SIZE - 1] = 0;
   }

   for (int d = 1; d < TILE_SIZE; d *= 2)
   {
      offset >>= 1;
      __syncthreads();
      if (tid < d)                     
      { 
         int ai = offset*(2*tid+1)-1;
         int bi = offset*(2*tid+2)-1;
         
         unsigned int t = temp[ai];
         temp[ai] = temp[bi];
         temp[bi] += t; 
      }
   }

   __syncthreads();
   g_odata[2*index] = temp[2*tid];
   g_odata[2*index+1] = temp[2*tid+1];
   
}

__global__ void addSumsToEachElement(unsigned int *g_odata, unsigned int *incr, int n)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int index = bid * blockDim.x + tid; 
  if((2*index + 1) < n)
  {
    g_odata[2*index] += incr[bid];
    g_odata[2*index + 1] += incr[bid];
  }
}
// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, unsigned int *sums, unsigned int *incr, 
                   int numElements, int padded_num_elementsIncrArray)
{
   int numBlocks = numElements/TILE_SIZE;

   dim3 dimGrid(numBlocks, 1, 1);
   dim3 dimBlock(BLOCK_SIZE, 1, 1);

   prescan<<<dimGrid, dimBlock>>>(outArray, inArray, sums, numElements);
   cudaDeviceSynchronize();  

   if(numBlocks != 1)
   {
     int numBlocksNew = padded_num_elementsIncrArray/TILE_SIZE;
     dim3 dimGridNew(numBlocksNew, 1, 1);
     dim3 dimBlockNew(BLOCK_SIZE, 1, 1);
     prescan<<<dimGridNew, dimBlockNew>>>(incr, sums, NULL, padded_num_elementsIncrArray); 
     cudaDeviceSynchronize();
   
     addSumsToEachElement<<<dimGrid, dimBlock>>>(outArray, incr, numElements);
   }
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
