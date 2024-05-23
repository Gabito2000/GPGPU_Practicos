// El objetivo del ejercicio es utilizar la memoria compartida para la privatización. Además se utilizará el
// patrón de reducción paralela. Antes de empezar, defina una matriz de enteros de 3840 × 2160 (4K 16:9) y
// adapte el kernel del histograma realizado en el práctico 2 para que trabaje sobre una matriz de enteros entre
// 0 y 255.
// • a) Desarrolle una variante donde cada bloque mantenga un histograma local en memoria compartida
// y, al final de la recorrida de la imagen, los datos del histograma local se impacten en el histograma
// global utilizando atomicAdd.
// • b) Desarrolle otra variante donde, una vez realizada la recorrida de la imagen, cada bloque impacte el
// histograma local en una fila de una “matriz de histogramas”. Luego, otro kernel debe realizar la suma
// por columnas de la matriz para obtener el histograma global.
// – Evite el acceso “no coalesced” eligiendo un tamaño de bloque adecuado.
// – Use el patrón de reducción visto en teórico para sumar cada segmento de una columna.
// – Guarde la suma parcial de cada bloque en la propia matriz de histogramas.
// – Invoque nuevamente el kernel para sumar las sumas parciales del paso anterior hasta que se haya
// sumado toda la columna.
// – Evite acceder a los datos en memoria global de forma “no coalesced” eligiendo un tamaño de
// bloque adecuado.
// – Agregue algunas filas de 0s a la matriz para que el número de filas sea múltiplo del tamaño de
// bloque elegido.
// – Compare el desempeño de esta solución con la de la parte a) y la adaptada del práctico 2.

// Ejercicio 2
// #include <stdio.h>
// #include <stdlib.h>
// #include "cuda_runtime.h"
// #include <iostream>
// #include <string>

// #define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess) 
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }

// void read_file(const char*, int*);
// int get_text_length(const char * fname);

// #define A 15
// #define B 27
// #define M 256
// #define A_MMI_M -17

// #define BLOCK_SIZE 256
// #define NUM_BLOCKS 128

// __device__ int modulo(int a, int b){
//     int r = a % b;
//     r = (r < 0) ? r + b : r;
//     return r;
// }

// __global__ void decrypt_kernel(int *d_message, int length)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
    
//     for (int i = idx; i < length; i += stride) {
//         d_message[i] = modulo(A_MMI_M * (d_message[i] - B), M);
//     }
// }

// __global__ void count_occurrences_kernel(int *d_message, int length, int *d_occurrences)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
    
//     for (int i = idx; i < length; i += stride) {
//         atomicAdd(&d_occurrences[d_message[i]], 1);
//     }
// }

// void print_cuda_error(const std::string &message) {
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         std::cout << message << " : " << cudaGetErrorString(error) << "\n";
//         std::cout << "Error OUT \n";
//         exit(1);
//     }
// }

// int main(int argc, char *argv[])
// {
//     int *h_message;
//     int *d_message;
//     int *h_occurrences;
//     int *d_occurrences;
//     unsigned int size;
//     const char * fname;

//     if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
//     else
//         fname = argv[1];

//     int length = get_text_length(fname);

//     size = length * sizeof(int);

//     // reservar memoria para el mensaje
//     h_message = (int *)malloc(size);

//     // leo el archivo de la entrada
//     read_file(fname, h_message);

//     /* reservar memoria en la GPU */
//     CUDA_CHK(cudaMalloc((void**)&d_message, size));
//     print_cuda_error("Error al reservar memoria en la GPU para el mensaje");

//     CUDA_CHK(cudaMalloc((void**)&d_occurrences, M * sizeof(int)));
//     print_cuda_error("Error al reservar memoria en la GPU para las ocurrencias");

//     // Inicializar el vector de ocurrencias con ceros
//     CUDA_CHK(cudaMemset(d_occurrences, 0, M * sizeof(int)));
//     print_cuda_error("Error al inicializar el vector de ocurrencias");

//     /* copiar los datos de entrada a la GPU */
//     CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));
//     print_cuda_error("Error al copiar los datos de entrada a la GPU");

//     /* Configurar la grilla y lanzar el kernel de desencriptación */
//     dim3 blockSize_decrypt(BLOCK_SIZE);
//     dim3 numBlocks_decrypt(NUM_BLOCKS);
    
//     decrypt_kernel<<<numBlocks_decrypt, blockSize_decrypt>>>(d_message, length);
//     print_cuda_error("Error al lanzar el kernel de desencriptación");

//     /* Configurar la grilla y lanzar el kernel de conteo de ocurrencias */
//     dim3 blockSize_count(BLOCK_SIZE);
//     dim3 numBlocks_count(NUM_BLOCKS);
    
//     count_occurrences_kernel<<<numBlocks_count, blockSize_count>>>(d_message, length, d_occurrences);
//     print_cuda_error("Error al lanzar el kernel de conteo de ocurrencias");

//     /* Retornar los datos de las ocurrencias a la CPU */
//     h_occurrences = (int *)malloc(M * sizeof(int));
//     CUDA_CHK(cudaMemcpy(h_occurrences, d_occurrences, M * sizeof(int), cudaMemcpyDeviceToHost));
//     print_cuda_error("Error al copiar los datos de las ocurrencias a la CPU");

//     // Imprimir las ocurrencias de cada caracter
//     for (int i = 0; i < M; i++) {
//         if (h_occurrences[i] > 0) {
//             printf("Caracter '%c': %d ocurrencias\n", (char)i, h_occurrences[i]);
//         }
//     }

//     // Liberar la memoria en la GPU
//     CUDA_CHK(cudaFree(d_message));
//     CUDA_CHK(cudaFree(d_occurrences));

//     // Liberar la memoria en la CPU
//     free(h_message);
//     free(h_occurrences);

//     return 0;
// }

// int get_text_length(const char * fname)
// {
//     FILE *f = NULL;
//     f = fopen(fname, "r"); //read and binary flags

//     size_t pos = ftell(f);    
//     fseek(f, 0, SEEK_END);    
//     size_t length = ftell(f); 
//     fseek(f, pos, SEEK_SET);  

//     fclose(f);

//     return length;
// }

// void read_file(const char * fname, int* input)
// {
//     FILE *f = NULL;
//     f = fopen(fname, "r"); //read and binary flags
//     if (f == NULL){
//         fprintf(stderr, "Error: Could not find %s file \n", fname);
//         exit(1);
//     }

//     int c; 
//     while ((c = getc(f)) != EOF) {
//         *(input++) = c;
//     }

//     fclose(f);
// }

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#define WIDTH 3840
#define HEIGHT 2160

#define HISTO_SIZE 256

#define BLOCK_SIZE_x 1024
#define BLOCK_SIZE_y 1

#define ITERATIONS 100

__global__ void count_occurrences_kernel(int *d_image, int *d_occurrences)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < length; i += stride) {
        atomicAdd(&d_occurrences[d_image[i]], 1);
    }
}

int main_viejo () {
    int *h_image;
    int *d_image;
    int *h_occurrences;
    int *d_occurrences;
    unsigned int size;

    // reservar memoria para la imagen
    h_image = (int *)malloc(WIDTH * HEIGHT * sizeof(int));

    // Inicializar la imagen
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_image[i] = rand() % HISTO_SIZE;
    }

    // reservar memoria en la GPU
    cudaMalloc((void**)&d_image, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc((void**)&d_occurrences, HISTO_SIZE * sizeof(int));

    // copiar los datos de entrada a la GPU
    cudaMemcpy(d_image, h_image, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

    // Inicializar el vector de ocurrencias con ceros
    cudaMemset(d_occurrences, 0, HISTO_SIZE * sizeof(int));

    // Configurar la grilla y lanzar el kernel de conteo de ocurrencias
    dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y);
    dim3 numBlocks(WIDTH * HEIGHT / BLOCK_SIZE_x, 1);
    
    count_occurrences_kernel<<<numBlocks, blockSize>>>(d_image, d_occurrences);

    // Retornar los datos de las ocurrencias a la CPU
    h_occurrences = (int *)malloc(HISTO_SIZE * sizeof(int));
    cudaMemcpy(h_occurrences, d_occurrences, HISTO_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir las ocurrencias de cada caracter
    for (int i = 0; i < HISTO_SIZE; i++) {
        if (h_occurrences[i] > 0) {
            printf("Caracter '%c': %d ocurrencias\n", (char)i, h_occurrences[i]);
        }
    }

    // Liberar la memoria en la GPU
    cudaFree(d_image);
    cudaFree(d_occurrences);

    // Liberar la memoria en la CPU
    free(h_image);
    free(h_occurrences);

    return 0;
}

int main( int argc, char **argv ) {
    for (int i = 0; i < ITERATIONS; i++) {
        main_viejo();
    }
    return 0;
}




