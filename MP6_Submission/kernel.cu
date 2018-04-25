#include <stdio.h>
#define BLOCK_SIZE 1024


__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
    unsigned int *csrColIdx, float *csrData, float *inVector, 
    float *outVector) {

    // INSERT KERNEL CODE HERE
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < dim)
    {
       float dot = 0;
       int row_start = csrRowPtr[row];
       int row_end = csrRowPtr[row + 1];
       for(int j=row_start; j < row_end; j++){
          dot += csrData[j] * inVector[csrColIdx[j]];
       }
       outVector[row] = dot;
    } 
}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {

    // INSERT KERNEL CODE HERE
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < dim)
    {
       float dot = 0;
       for(int j=0; j < jdsRowNNZ[row]; j++){
          dot += jdsData[row + jdsColStartIdx[j]] * inVector[jdsColIdx[row + jdsColStartIdx[j]]];
       }
       outVector[jdsRowPerm[row]] = dot;
    }

}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float *csrData, float *inVector, float *outVector) {

    // INSERT CODE HERE
    int numBlocks = dim/BLOCK_SIZE;
    if(dim%BLOCK_SIZE) numBlocks++;

    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    spmv_csr_kernel<<<dimGrid, dimBlock>>>(dim, csrRowPtr, csrColIdx, csrData, inVector, outVector); 
    cudaDeviceSynchronize(); 
}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
    float* inVector, float *outVector) {

    // INSERT CODE HERE
    int numBlocks = dim/BLOCK_SIZE;
    if(dim%BLOCK_SIZE) numBlocks++;

    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    
    spmv_jds_kernel<<<dimGrid, dimBlock>>>(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx,
                                            jdsColIdx, jdsData, inVector, outVector);
    cudaDeviceSynchronize();
}






