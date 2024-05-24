#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#define WIDTH 3840
#define WIDTH_WITH_PADDING 4096
#define HEIGHT 2160
#define HISTO_SIZE 256
#define BLOCK_SIZE 1024

#define ITERATIONS 100

__global__ void histo_kernel_shared_part(int *d_image, int *d_histogram_partial, int matrixSize) {
    __shared__ int histo_private[HISTO_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < HISTO_SIZE) {
        histo_private[tid] = 0;
    }

    __syncthreads();
    
    for (int i = gid; i < matrixSize; i += blockDim.x * gridDim.x) {
        atomicAdd(&histo_private[d_image[i]], 1);
    }

    __syncthreads();
    
    if (tid < HISTO_SIZE) {
        atomicAdd(&d_histogram_partial[blockIdx.x * HISTO_SIZE + tid], histo_private[tid]);
    }
}

__global__ void add_histo(int *d_histogram_partial, int *d_histogram) {
    __shared__  int histo_private[BLOCK_SIZE];

    if (threadIdx.x < BLOCK_SIZE) {
        histo_private[threadIdx.x] = 0;
    }

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&histo_private[tid % HISTO_SIZE], d_histogram_partial[tid]);

    __syncthreads();

    atomicAdd(&d_histogram[threadIdx.x], histo_private[threadIdx.x]);
}

int main_nuevo() {
    int *h_image = (int *)malloc(WIDTH_WITH_PADDING * HEIGHT * sizeof(int));
    int *d_image, *d_histogram, *d_histogram_partial;
    int h_histogram[HISTO_SIZE] = {0};

    // Inicializar imagen con valores aleatorios
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_image[i] = i % HISTO_SIZE;
    }

    for (int i = WIDTH * HEIGHT; i < WIDTH_WITH_PADDING * HEIGHT; i++) {
        h_image[i] = 0;
    }

    int NUM_BLOCKS = (WIDTH_WITH_PADDING * HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void**)&d_image, WIDTH_WITH_PADDING * HEIGHT * sizeof(int));
    cudaMalloc((void**)&d_histogram, HISTO_SIZE * sizeof(int));
    cudaMalloc((void**)&d_histogram_partial, HISTO_SIZE * NUM_BLOCKS * sizeof(int));
    
    cudaMemcpy(d_image, h_image, WIDTH_WITH_PADDING * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, HISTO_SIZE * sizeof(int));
    cudaMemset(d_histogram_partial, 0, HISTO_SIZE * NUM_BLOCKS * sizeof(int));

    dim3 blockSize(BLOCK_SIZE);
    dim3 numBlocks(NUM_BLOCKS); 

    histo_kernel_shared_part<<<numBlocks, blockSize>>>(d_image, d_histogram_partial, WIDTH_WITH_PADDING * HEIGHT);
    cudaDeviceSynchronize();

    NUM_BLOCKS = NUM_BLOCKS * HISTO_SIZE / BLOCK_SIZE;

    add_histo<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_histogram_partial, d_histogram);
    cudaDeviceSynchronize();

    cudaMemcpy(h_histogram, d_histogram, HISTO_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    h_histogram[0] =  h_histogram[0] - (WIDTH_WITH_PADDING - WIDTH) * HEIGHT; // Remove padding
    for (int i = 0; i < HISTO_SIZE; i++) {
        printf("Bin %d: %d\n", i, h_histogram[i]);
    }
    printf("CUDA ERROR: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaFree(d_image);
    cudaFree(d_histogram);
    cudaFree(d_histogram_partial);
    free(h_image);

    return 0;
}

int main(int argc, char **argv) {
    for (int i = 0; i < ITERATIONS; i++) {
        main_nuevo();
    }
    return 0;
}
