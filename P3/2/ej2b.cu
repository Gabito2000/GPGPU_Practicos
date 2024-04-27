#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void sumNeighborElementAligned(int *matrix, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int currentElementIndex = row * width + col;
        int neighborElementIndex = row * width + (col + 4);

        if (col + 4 < width) {
            atomicAdd(&matrix[currentElementIndex], matrix[neighborElementIndex]);
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




/*El patrón de acceso a memoria en el kernel proporcionado en el ejercicio 2a puede no ser óptimo debido a que cada hilo accede a elementos de la matriz de forma desalineada, es decir, los elementos (x, y) y (x+4, y) no están contiguos en memoria. Esto puede resultar en un mayor número de transacciones de memoria y una menor eficiencia en la transferencia de datos entre la GPU y la memoria global.

Para mitigar el efecto del acceso desalineado a la memoria global, podemos reorganizar el acceso de los hilos de manera que accedan a la memoria de forma coalescente. Esto implica que los hilos accedan a elementos de la matriz en patrones de acceso consecutivos, lo que permite agrupar múltiples accesos en una única transacción de memoria.

Una forma de lograr esto es mediante el reordenamiento de los bloques y los hilos de manera que los hilos accedan a elementos contiguos de la matriz en la dimensión que produce coalescencia. Por ejemplo, si tenemos una matriz almacenada en un formato de fila mayor (row-major order), podemos organizar los hilos de manera que cada hilo acceda a una fila completa de la matriz.*/


/*En este kernel modificado, cada hilo accede a elementos en la misma fila de manera coalescente. Además, utilizamos atomicAdd para evitar condiciones de carrera al actualizar los elementos de la matriz.

Para configurar los parámetros de la grilla y los bloques de manera que mantengan este patrón de acceso coalescente, podemos utilizar un tamaño de bloque de (32, 32) y ajustar la dimensión de la grilla según el tamaño de la matriz.*/