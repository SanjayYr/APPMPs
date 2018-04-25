/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  //Multiply the two matrices
  // Calcullate row index of matrix of M and P
  unsigned int Row = threadIdx.y + (blockIdx.y * blockDim.y);
  
  // Calcullate column index of matrix N and P   
  unsigned int Col = threadIdx.x + (blockIdx.x * blockDim.x);

  if((Row < MATRIX_SIZE) && (Col < MATRIX_SIZE))  // Since Width=MATRIX_SIZE
  {
     float Pvalue = 0;
     // Each thread computes one element of block sub-matrix

     for(unsigned int k=0; k < MATRIX_SIZE; ++k)        // Again Width=MATRIX_SIZE here
     {
        Pvalue += M.elements[Row * MATRIX_SIZE + k] * N.elements[k * MATRIX_SIZE + Col];   
        // Both M and N elements are stored in Row-major layout
     }
     P.elements[Row*MATRIX_SIZE + Col] = Pvalue; 
  } 
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
