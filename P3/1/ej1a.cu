#include <stdio.h>

#define BLOCK_SIZE 32
#define ITERATIONS 10

__global__ void transposeMatrix(int *inputMatrix, int *outputMatrix, int width, int height) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;

    if (globalIdx < width && globalIdy < height) {
        int index_input = globalIdy * width + globalIdx;
        int index_output = globalIdx * height + globalIdy;

        outputMatrix[index_output] = inputMatrix[index_input];
    }
}

int main_original() {
    // Define matrix dimensions
    int width = 1024; 
    int height = 1024;
    int matrixSize = width * height;

    // Allocate memory for the matrices on the host
    int *h_inputMatrix = (int*)malloc(matrixSize * sizeof(int));
    int *h_outputMatrix = (int*)malloc(matrixSize * sizeof(int));

    // Initialize input matrix (you can do this randomly or with specific values)
    for (int i = 0; i < matrixSize; ++i) {
        h_inputMatrix[i] = i;
    }

    // Print input matrix
    // printf("Input matrix:\n");
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         printf("%d ", h_inputMatrix[i * width + j]);
    //     }
    //     printf("\n");
    // }

    // Allocate memory for the matrices on the device
    int *d_inputMatrix, *d_outputMatrix;
    cudaMalloc((void**)&d_inputMatrix, matrixSize * sizeof(int));
    cudaMalloc((void**)&d_outputMatrix, matrixSize * sizeof(int));

    // Copy input matrix from host to device
    cudaMemcpy(d_inputMatrix, h_inputMatrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    transposeMatrix<<<gridSize, blockSize>>>(d_inputMatrix, d_outputMatrix, width, height);

    // Copy result back to host
    cudaMemcpy(h_outputMatrix, d_outputMatrix, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output matrix
    // printf("Output matrix:\n");
    // for (int i = 0; i < width; ++i) {
    //     for (int j = 0; j < height; ++j) {
    //         printf("%d ", h_outputMatrix[i * height + j]);
    //     }
    //     printf("\n");
    // }

    // Free device memory
    cudaFree(d_inputMatrix);
    cudaFree(d_outputMatrix);

    // Free host memory
    free(h_inputMatrix);
    free(h_outputMatrix);

    return 0;
}

int main() {
    for (int i = 0; i < ITERATIONS; i++) {
        main_original();
}