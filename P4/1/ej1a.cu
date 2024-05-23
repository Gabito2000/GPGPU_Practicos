#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#define ITERATIONS 100
#define WARPSIZE 32
#define TILE_DIM 32
#define BLOCK_ROWS 8

uint64_t get_nanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}

__global__ void transposeMatrix(int *inputMatrix, int *outputMatrix) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = inputMatrix[(y + j) * width + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        outputMatrix[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

int main_nuevo(int argc, char **argv) {
    int width = 1024; 
    int height = 1024;
    int matrixSize = width * height;

    int *h_i = (int*)malloc(matrixSize * sizeof(int));
    int *h_o = (int*)malloc(matrixSize * sizeof(int));

    for (int i = 0; i < matrixSize; ++i) {
        h_i[i] = i;
    }

    int *d_i, *d_o;
    cudaMalloc((void**)&d_i, matrixSize * sizeof(int));
    cudaMalloc((void**)&d_o, matrixSize * sizeof(int));

    cudaMemcpy(d_i, h_i, matrixSize * sizeof(int), cudaMemcpyHostToDevice);


    dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    dim3 numBlocks(width / TILE_DIM, height / TILE_DIM);

    transposeMatrix<<<numBlocks, blockSize>>>(d_i, d_o);

    cudaMemcpy(h_o, d_o, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_i);
    cudaFree(d_o);

    free(h_i);
    free(h_o);

    printf("CUDA ERROR: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}



int main(int argc, char **argv) {

    uint64_t start = get_nanoseconds();
    for (int i = 0; i < ITERATIONS; i++) {
        main_nuevo(argc, argv);
        cudaDeviceReset();
    }
    uint64_t end = get_nanoseconds();
    printf("Time: %lu ns\n", (end - start) / ITERATIONS);

    return 0;
}