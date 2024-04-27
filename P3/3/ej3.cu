__global__ void matrixVectorMultiplication(int *A, int *v, int *x, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows) {
        int sum = 0;
        for (int i = 0; i < 256; ++i) {
            sum += A[row * 256 + i] * v[i];
        }
        x[row] = sum;
    }
}


int main() {
    // Definir dimensiones de la matriz A y del vector v
    int numRows = 10240;
    int numCols = 256;
    int matrixSize = numRows * numCols;
    int vectorSize = numCols;

    // Reservar memoria en el host
    int *h_A = (int*)malloc(matrixSize * sizeof(int));
    int *h_v = (int*)malloc(vectorSize * sizeof(int));
    int *h_x = (int*)malloc(numRows * sizeof(int));

    // Inicializar matriz A y vector v (puedes hacerlo aleatoriamente o con valores específicos)
    for (int i = 0; i < matrixSize; ++i) {
        h_A[i] = i % 10; // ejemplo de inicialización
    }
    for (int i = 0; i < vectorSize; ++i) {
        h_v[i] = i % 5; // ejemplo de inicialización
    }

    // Reservar memoria en el dispositivo
    int *d_A, *d_v, *d_x;
    cudaMalloc((void**)&d_A, matrixSize * sizeof(int));
    cudaMalloc((void**)&d_v, vectorSize * sizeof(int));
    cudaMalloc((void**)&d_x, numRows * sizeof(int));

    // Copiar matriz A y vector v desde el host al dispositivo
    cudaMemcpy(d_A, h_A, matrixSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    // Definir tamaño de bloque y de la grilla
    int blockSize = 256; // Utilizamos el mismo tamaño de bloque que el número de columnas de la matriz A
    int gridSize = (numRows + blockSize - 1) / blockSize;

    // Registrar el tiempo de ejecución
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    // Ejecutar kernel
    matrixVectorMultiplication<<<gridSize, blockSize>>>(d_A, d_v, d_x, numRows);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución del kernel: %f ms\n", milliseconds);

    // Copiar vector resultado x desde el dispositivo al host
    cudaMemcpy(h_x, d_x, numRows * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria
    free(h_A);
    free(h_v);
    free(h_x);
    cudaFree(d_A);
    cudaFree(d_v);
    cudaFree(d_x);

    return 0;
}
