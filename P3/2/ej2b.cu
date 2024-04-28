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
    // Definir dimensiones de la matriz
    int width = 10;
    int height = 10;
    int matrixSize = width * height;

    // Reservar memoria en el host
    int *h_matrix = (int*)malloc(matrixSize * sizeof(int));

    // Inicializar matriz (puedes hacerlo aleatoriamente o con valores específicos)
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

    // Reservar memoria en el dispositivo
    int *d_matrix;
    cudaMalloc((void**)&d_matrix, matrixSize * sizeof(int));

    // Copiar matriz desde el host al dispositivo
    cudaMemcpy(d_matrix, h_matrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice);

    // Definir dimensiones de la grilla y tamaño de bloque
    dim3 blockSize(4,4);
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
    // printf("Tiempo de ejecución del kernel: %f ms\n", milliseconds);

    // Copiar matriz resultante desde el dispositivo al host
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

/*Para mitigar el efecto del acceso desalineado a la memoria global, podemos ajustar los parámetros de la grilla y el bloque para favorecer
 un patrón de acceso a memoria más coalescente. Esto implica cambiar el tamaño de la grilla y el bloque para que los hilos accedan a elementos
  de la matriz de manera más coherente.

Por ejemplo, podemos reducir el tamaño del bloque para que menos hilos estén accediendo a la memoria al mismo tiempo, lo que puede ayudar a
 reducir la fragmentación. Sin embargo, esta optimización puede reducir el grado de paralelismo, por lo que es importante encontrar un equilibrio
  adecuado.*/

int main() {
    for (int i = 0; i < ITERATIONS; i++) {
        main_original();
}
