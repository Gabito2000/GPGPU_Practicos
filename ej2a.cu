#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void addNeighborElement(int *matrix, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int currentElementIndex = y * width + x;
        int neighborElementIndex = y * width + (x + 4);

        if (x + 4 < width) {
            matrix[currentElementIndex] += matrix[neighborElementIndex];
        }
    }
}


int main() {
    // Definir dimensiones de la matriz
    int width = 1024;
    int height = 1024;
    int matrixSize = width * height;

    // Reservar memoria en el host
    int *h_matrix = (int*)malloc(matrixSize * sizeof(int));

    // Inicializar matriz (puedes hacerlo aleatoriamente o con valores específicos)
    for (int i = 0; i < matrixSize; ++i) {
        h_matrix[i] = i;
    }

    // Reservar memoria en el dispositivo
    int *d_matrix;
    cudaMalloc((void**)&d_matrix, matrixSize * sizeof(int));

    // Copiar matriz desde el host al dispositivo
    cudaMemcpy(d_matrix, h_matrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice);

    // Definir dimensiones de la grilla y tamaño de bloque
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Registrar el tiempo de ejecución
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    // Ejecutar kernel
    addNeighborElement<<<gridSize, blockSize>>>(d_matrix, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución del kernel: %f ms\n", milliseconds);

    // Copiar matriz resultante desde el dispositivo al host
    cudaMemcpy(h_matrix, d_matrix, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria
    free(h_matrix);
    cudaFree(d_matrix);

    return 0;
}
