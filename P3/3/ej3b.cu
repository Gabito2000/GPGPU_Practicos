#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define ITERATIONS 10
#define TILE_WIDTH 16

__global__ void matrixVectorMultiplication(int *A, int *v, int *x, int numRows) {
    __shared__ int shared_v[256];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    
    if (tx < 256)
        shared_v[tx] = v[tx];
    
    __syncthreads();
    
    int sum = 0;
    for (int j = 0; j < 256; j += TILE_WIDTH) {
        int tile_start = i * 256 + j;
        int tile_end = min(tile_start + TILE_WIDTH, 256); //min es cuda operation así que no debería de dar porblemas cuando se trabaja con la gpu
        for (int k = tile_start; k < tile_end; ++k) {
            sum += A[k] * shared_v[k - i * 256];
        }
    }
    
    if (i < numRows)
        atomicAdd(&x[i], sum);
}




int main_original() {
    int numRows = 10240;
    int numCols = 256;
    int matrixSize = numRows * numCols;
    int vectorSize = numCols;

    int *h_A = (int*)malloc(matrixSize * sizeof(int));
    int *h_v = (int*)malloc(vectorSize * sizeof(int));
    int *h_x = (int*)malloc(numRows * sizeof(int));

    for (int i = 0; i < matrixSize; ++i) {
        h_A[i] = i;
    }
    for (int i = 0; i < vectorSize; ++i) {
        h_v[i] = i;
    }

    int *d_A, *d_v, *d_x;
    cudaMalloc((void**)&d_A, matrixSize * sizeof(int));
    cudaMalloc((void**)&d_v, vectorSize * sizeof(int));
    cudaMalloc((void**)&d_x, numRows * sizeof(int));

    cudaMemcpy(d_A, h_A, matrixSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    int gridSize = (numRows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    matrixVectorMultiplication<<<gridSize, BLOCK_SIZE>>>(d_A, d_v, d_x, numRows);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Tiempo de ejecución del kernel: %f ms\n", milliseconds);

    cudaMemcpy(h_x, d_x, numRows * sizeof(int), cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_v);
    free(h_x);
    cudaFree(d_A);
    cudaFree(d_v);
    cudaFree(d_x);

    // printf("Errors: \n");
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    return 0;
}


int main() {
    for (int i = 0; i < ITERATIONS; i++) {
        main_original();
    }
}