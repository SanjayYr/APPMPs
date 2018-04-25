#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(unsigned int *g_data, int n)
{
  __shared__ unsigned int partialSum[2*BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int start = 2*blockDim.x*blockIdx.x;

  if((start + tid) < n)
  {
    partialSum[tid] = g_data[start + tid];
  }
  else
  {
    partialSum[tid] = 0;
  }

  if((start + blockDim.x + tid) < n)
  {
    partialSum[blockDim.x + tid] = g_data[start + blockDim.x + tid];
  }
  else
  {
    partialSum[blockDim.x + tid] = 0;
  }
  // First implementation. Tried testing both versions as pointed out by Professor in class
  /*
  for (unsigned int stride = 1;
          stride <= blockDim.x;  stride *= 2)
  {
     __syncthreads();
     if (tid % stride == 0)
     {
       partialSum[2*tid]+= partialSum[2*tid+stride];
     }
  } */

  for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1)
  {
     __syncthreads();
     if (tid < stride)
     {
       partialSum[tid] += partialSum[tid + stride];
     }
  }
  *(g_data + (blockIdx.x * (2*BLOCK_SIZE))) = partialSum[0];
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
