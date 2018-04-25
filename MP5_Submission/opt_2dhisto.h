#ifndef OPT_KERNEL
#define OPT_KERNEL


/* Include below the function headers of any other functions that you implement */

void opt_2dhisto( uint32_t *input[], size_t height,
                    size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH] );
void device_setup( uint32_t *input[], size_t height,
                    size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH] );
void device_copy_and_cleaup(uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

void* deviceAllocateCpy(uint32_t *in[], size_t height, size_t width );
void CopyFromDevice(void* D_host, void* D_device, size_t size);
void FreeDevice(void* D_device);

#endif
