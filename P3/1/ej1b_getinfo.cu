#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


//get the cuda warp size
int getWarpSize() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.warpSize;
}

int main(){
    printf("Warp size: %d\n", getWarpSize());
    return 0;
}