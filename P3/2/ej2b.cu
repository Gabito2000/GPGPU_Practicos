#include <stdio.h>

#define BLOCK_SIZE 64
#define ITERATIONS 10

__global__ void addNeighborElement(int *matrix, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int currentElementIndex = y * width + x;
    int neighborElementIndex =y * width + (x + 4);

    if (x >= width) return;
    atomicAdd(&matrix[currentElementIndex], matrix[neighborElementIndex] );
}


int main_original() {
    int width = 10;
    int height = 10;
    int matrixSize = width * height;

    int *h_matrix = (int*)malloc(matrixSize * sizeof(int));

    for (int i = 0; i < matrixSize; ++i) {
        h_matrix[i] = i;
    }
    // printf("Matriz original: \n");
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         printf("%d ", h_matrix[i * width + j]);
    //     }
    //     printf("\n");
    // }
    // printf("--------\n");

    int *d_matrix;
    cudaMalloc((void**)&d_matrix, matrixSize * sizeof(int));

    cudaMemcpy(d_matrix, h_matrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(4,4);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    addNeighborElement<<<gridSize, blockSize>>>(d_matrix, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Tiempo de ejecución del kernel: %f ms\n", milliseconds);

    cudaMemcpy(h_matrix, d_matrix, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);

    // printf("Resultado: \n");
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         printf("%d ", h_matrix[i * width + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // Liberar memoria
    free(h_matrix);
    cudaFree(d_matrix);

    return 0;
}

int main() {
    for (int i = 0; i < ITERATIONS; i++) {
        main_original();
    }
}
