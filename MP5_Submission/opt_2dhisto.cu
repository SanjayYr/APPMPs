#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"
#define BLOCK_SIZE 1024
#define GRID_SIZE 1024

void device_copy_and_cleaup(uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

__global__ void kernelHistogram(uint32_t *input, size_t height, size_t width,
                            uint32_t *bins);
__global__ void kernelBins32to8(uint32_t *input_d, uint8_t *output, int size);

void CopyFromDevice(void* D_host, void* D_device, size_t size);
void FreeDevice(void* D_device);

uint32_t *bins_device_32;
uint32_t *input_d;
uint8_t *bins_device_8;

void device_setup(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]){

    unsigned int inpSize = height*width*sizeof(uint32_t);
    cudaMalloc((void**)&bins_device_32, (int)(HISTO_WIDTH*HISTO_HEIGHT)*sizeof(uint32_t));
    cudaMemset(bins_device_32, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));    
    
    cudaMalloc((void**)&bins_device_8, (int)(HISTO_WIDTH*HISTO_HEIGHT)*sizeof(uint8_t));
    cudaMemset(bins_device_8, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
     
    cudaMalloc((void**)&input_d, inpSize);
    for(int i = 0; i < INPUT_HEIGHT; ++i)
    {
      cudaMemcpy((input_d + i*INPUT_WIDTH), input[i], 
                      INPUT_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }	
    
    cudaDeviceSynchronize();
}

void opt_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]){
    dim3 dimGrid(GRID_SIZE, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    kernelHistogram<<<dimGrid, dimBlock>>>(input_d, height, width, bins_device_32);
    cudaDeviceSynchronize();
    device_copy_and_cleaup(bins);
    cudaDeviceSynchronize();
}

void device_copy_and_cleaup(uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
   kernelBins32to8<<< ceil((float)(HISTO_HEIGHT*HISTO_WIDTH)/(float)(
                                    BLOCK_SIZE)), BLOCK_SIZE >>>(
                                        bins_device_32, bins_device_8,
                                        HISTO_WIDTH*HISTO_HEIGHT);
   cudaDeviceSynchronize();
   CopyFromDevice(bins, bins_device_8, HISTO_WIDTH*HISTO_HEIGHT);
   cudaDeviceSynchronize(); 
   FreeDevice(bins_device_8);
   FreeDevice(bins_device_32);
   FreeDevice(input_d);
}

__global__ void kernelHistogram(uint32_t *input, size_t height, size_t width, uint32_t *bins){

   __shared__ int histo_private[HISTO_WIDTH];

   if (threadIdx.x < HISTO_WIDTH) histo_private[threadIdx.x] = 0;
   __syncthreads();

   int index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   while (index < width*height) {
      if(histo_private[input[index]]<255)
        atomicAdd( &(histo_private[input[index]]), 1);
      index += stride;
   }
    __syncthreads();

   if (threadIdx.x < HISTO_WIDTH ){
      atomicAdd( &(bins[threadIdx.x]), histo_private[threadIdx.x] );
   }

}

__global__ void kernelBins32to8(uint32_t *input_d, uint8_t *output, int size){
   int i = threadIdx.x + blockDim.x*blockIdx.x;
   if(i< size){
      output[i] = (uint8_t)((input_d[i] > 255 )? UINT8_MAX:input_d[i]);
   }
}

/* Include below the implementation of any other functions you need */
void CopyFromDevice(void* D_host, void* D_device, size_t size){
   cudaMemcpy(D_host, D_device, size, cudaMemcpyDeviceToHost);
}

void FreeDevice(void* D_device){
   cudaFree(D_device);
}

