#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#define ITERATIONS 10
#define WARPSIZE 32

uint64_t get_nanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}

__global__ void transposeMatrix(int *inputMatrix, int *outputMatrix, int width, int height) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;

    if (globalIdx < width && globalIdy < height) {
        int index_input = globalIdy * width + globalIdx;
        int index_output = globalIdx * height + globalIdy;

        outputMatrix[index_output] = inputMatrix[index_input];
    }
}

int main_original(int argc, char **argv) {
    int BLOCK_SIZE_x = 32;
    int BLOCK_SIZE_y = 1;

    int width = 1024; 
    int height = 1024;
    int matrixSize = width * height;

    int *h_i = (int*)malloc(matrixSize * sizeof(int));
    int *h_o = (int*)malloc(matrixSize * sizeof(int));

    for (int i = 0; i < matrixSize; ++i) {
        h_i[i] = i;
    }

    // Print input matrix
    // printf("Input matrix:\n");
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         printf("%d ", h_i[i * width + j]);
    //     }
    //     printf("\n");
    // }

    int *d_i, *d_o;
    cudaMalloc((void**)&d_i, matrixSize * sizeof(int));
    cudaMalloc((void**)&d_o, matrixSize * sizeof(int));

    cudaMemcpy(d_i, h_i, matrixSize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);
    dim3 numBlocks(32);

    transposeMatrix<<<numBlocks, blockSize>>>(d_i, d_o, width, height);

    cudaMemcpy(h_o, d_o, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output matrix
    // printf("Output matrix:\n");
    // for (int i = 0; i < width; ++i) {
    //     for (int j = 0; j < height; ++j) {
    //         printf("%d ", h_o[i * height + j]);
    //     }
    //     printf("\n");
    // }

    cudaFree(d_i);
    cudaFree(d_o);

    free(h_i);
    free(h_o);

    printf("CUDA ERROR: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}

int main(int argc, char **argv) {
    uint64_t start, end;
    start = get_nanoseconds();

    for (int i = 0; i < ITERATIONS; i++) {
        main_original( argc, argv );
        cudaDeviceSynchronize();
    }

    end = get_nanoseconds();
    printf("Time: %lu ns\n", end - start);
}