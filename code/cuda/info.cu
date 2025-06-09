
#include <stdio.h>


int main() {
    int count;
    cudaError_t e;
    
    e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        exit(1);
    }
    printf("found %d devices\n", count);

    int device;
    e = cudaGetDevice(&device);
    if (e != cudaSuccess) {
        exit(2);
    }
    printf("using device %d\n", device);

    cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, device);
    if (e != cudaSuccess) {
        exit(3);
    }
    printf("using a %s\n", prop.name);
    printf("%d multiprocessors\n", prop.multiProcessorCount);
    printf("max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("warp size: %d\n", prop.warpSize);
    printf("global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0*1024*1024));
    printf("L2 cache: %.2f MB\n", prop.l2CacheSize / (1024.0*1024));
    printf("max shared memory per block: %d KB\n", (int) (prop.sharedMemPerBlock / 1024.0));
    printf("max shared memory per SM: %d KB\n", (int) (prop.sharedMemPerMultiprocessor / 1024.0));
    printf("compute capability: %d.%d\n", prop.major, prop.minor);
    
}