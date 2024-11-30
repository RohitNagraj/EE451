#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("Error: Failed to get CUDA device count: %s\n", 
               cudaGetErrorString(error));
        return 1;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("\nDevice %d Properties:\n", i);
        printf("  Device Name: %s\n", deviceProp.name);
        printf("  Compute Capability: %d.%d\n", 
               deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Max Threads per Block: %d\n", 
               deviceProp.maxThreadsPerBlock);
        printf("  Max Threads Dimensions: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Size: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  asyncEngineCount: %d\n", deviceProp.asyncEngineCount);
    }
    
    return 0;
}
