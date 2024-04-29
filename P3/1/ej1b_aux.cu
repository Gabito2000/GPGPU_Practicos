#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//uint64_t
#include <stdint.h>

#define BLOCK_SIZE 32
#define ITERATIONS 100


__global__ void transposeMatrix(int *inputMatrix, int *outputMatrix, int width, int height) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;

    if (globalIdx < width && globalIdy < height) {
        int index_input = globalIdy * width + globalIdx;
        int index_output = globalIdx * height + globalIdy;

        outputMatrix[index_output] = inputMatrix[index_input];
    }
}

uint64_t get_nanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}


int main() {
    //define block sizes
    int block_sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};


    // Define matrix dimensions
    int width = 1024; 
    int height = 1024;
    int matrixSize = width * height;
    
    
    // printf("Time for block size, time in ns, ITERATIONS: \n");

    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++) {
        int *h_inputMatrix = (int*)malloc(matrixSize * sizeof(int));
        for (int aux = 0; aux < matrixSize; ++aux) {
            h_inputMatrix[aux] = aux;
        }
        
        // declare acc time
        uint64_t acc_time = 0;

        for (int j = 0; j < ITERATIONS; j++) {
            // Allocate memory for the matrices on the host
            int *h_outputMatrix = (int*)malloc(matrixSize * sizeof(int));

            // Allocate memory for the matrices on the device
            int *d_inputMatrix, *d_outputMatrix;
            cudaMalloc((void**)&d_inputMatrix, matrixSize * sizeof(int));
            cudaMalloc((void**)&d_outputMatrix, matrixSize * sizeof(int));

            // Copy input matrix from host to device
            cudaMemcpy(d_inputMatrix, h_inputMatrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice);

            // Define grid and block dimensions
            // // dim3 blockSize(BLOCK_SIZE);
            // // dim3 numBlocks(NUM_BLOCKS);
            dim3 blockSize(block_sizes[i]);
            dim3 numBlocks(32);

            // Launch kernel
            uint64_t start = get_nanoseconds();
            transposeMatrix<<<numBlocks, blockSize>>>(d_inputMatrix, d_outputMatrix, width, height);
            uint64_t end = get_nanoseconds();
            acc_time += end - start;
            // Copy result back to host
            cudaMemcpy(h_outputMatrix, d_outputMatrix, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(d_outputMatrix);

            // Free host memory
            cudaFree(d_inputMatrix);
            free(h_outputMatrix);

            cudaDeviceReset();
        }

        

        // Free host memory
        free(h_inputMatrix);
        printf("%d, %lu, %d\n", block_sizes[i], (acc_time) / ITERATIONS, ITERATIONS);
    }
    printf("cudaError: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}