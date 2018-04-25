/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

#define TILE_WIDTH 32
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
   __shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
   __shared__ float N_s[TILE_WIDTH][TILE_WIDTH];

   int bx = blockIdx.x;    int tx = threadIdx.x;
   int by = blockIdx.y;    int ty = threadIdx.y;

   // Get the row and column indices of Pd elements to work on
   int Row = by * TILE_WIDTH + ty;
   int Col = bx * TILE_WIDTH + tx;

   int numOfPhases = M.width/TILE_WIDTH;
   if(M.width%TILE_WIDTH) numOfPhases++; // Case where Width is not a multiple of TILE_WIDTH

   float Pvalue = 0;
   // Loop over Md and Nd matrices to compute Pd elements
   for(int m = 0; m < numOfPhases; ++m)
   {
      // Load Md and Nd elements into shared memory M_s and N_s
      if(Row < M.height && ((m*TILE_WIDTH + tx) < M.width))
      {
         M_s[ty][tx] = M.elements[Row*M.width + m*TILE_WIDTH + tx];
      }

      if(Col < N.width && ((m*TILE_WIDTH + ty) < N.height))
      {
         N_s[ty][tx] = N.elements[(m*TILE_WIDTH + ty)*N.width + Col];
      }
      __syncthreads();

      for(int k = 0; k < TILE_WIDTH; ++k)
      {
         if((Row < M.height && ((m*TILE_WIDTH + k) < M.width)) &&
            (Col < N.width && ((m*TILE_WIDTH + k) < N.height)))
         {
           Pvalue += M_s[ty][k] * N_s[k][tx];
         }
      }
      __syncthreads();
   }
   if((Row < P.height) && (Col < P.width))
   {
      P.elements[Row*P.width + Col] = Pvalue;
   }

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
