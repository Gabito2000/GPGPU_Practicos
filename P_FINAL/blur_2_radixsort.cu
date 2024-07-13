#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvToolsExt.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <chrono>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <thrust/iterator/constant_iterator.h>
#define MAX_DIGITS 32 // Assuming 32-bit integers


using namespace std;


// Anexo A: Radix Sort
// Uno de los algoritmos de ordenamiento más eficientes para ordenar claves cortas en
// procesadores paralelos es el radix sort. El algoritmo comienza considerando el primer bit de
// cada clave, empezando por el bit menos significativo. Utilizando este bit se particiona el
// conjunto de claves de forma que todas las claves que tengan un 0 en ese bit se ubiquen antes
// que las claves que tienen el bit en 1, manteniendo el orden relativo de las claves con mismo
// valor de bit. Una vez completado este paso se hace lo mismo para cada uno de los bits de la
// clave hasta completar todos sus bits.
// Definimos la primitiva split(input, n) como la operación que ordena el arreglo input de acuerdo
// al valor b del bit n de cada elemento. Para implementar en GPU dicha primitiva se procederá
// de la siguiente manera:
// • En un arreglo temporal e almacenar el valor de not b para cada posición i de input.
// • Computar la suma prefija (exclusive scan) del arreglo. Ahora cada posición del arreglo
// contiene la cantidad de f de elementos de input con b=0 que hay antes que esa posición.
// Para los elementos con b=0, esta cantidad determina la posición en el arreglo de salida.
// El último elemento del arreglo de salida del scan contiene el total de posiciones con b=0
// (hay que sumar 1 a este valor si la última posición tiene b=0), denominada totalFalses.
// • Ahora se computa el índice de las posiciones con b=1 en el arreglo de salida. Para cada
// posición i, este índice será t = i - f + totalFalses.
// • Una vez obtenidos los índices anteriores se graba cada elemento de input en el arreglo
// de salida en la posición t o f dependiendo de si b es 1 o 0.
// Para implementar el algoritmo de radix sort utilizando la primitiva split simplemente debe
// inicializarse una máscara binaria para aislar el bit menos significativo, realizar el split del
// arreglo según ese bit, comprobar si el arreglo ya está ordenado y, si no lo está, hacer un shift
// a la izquierda de la máscara y volver a iterar. El procedimiento anterior se ejemplifica en la
// figura.

void radixSort_cpu(std::vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return;

    std::vector<int> output(n);
    std::vector<int> bitArray(n);
    std::vector<int> prefixSum(n);

    for (int bit = 0; bit < MAX_DIGITS; bit++) {
        int mask = 1 << bit;

        // Extract bit
        for (int i = 0; i < n; i++) {
            bitArray[i] = (arr[i] & mask) >> bit;
        }

        // Perform exclusive scan (prefix sum of not bit)
        prefixSum[0] = 0;
        for (int i = 1; i < n; i++) {
            prefixSum[i] = prefixSum[i - 1] + (1 - bitArray[i - 1]);
        }

        int totalFalses = prefixSum[n - 1] + (1 - bitArray[n - 1]);

        // Reorder
        std::fill(output.begin(), output.end(), 0); // Ensure the output vector is cleared

        for (int i = 0; i < n; i++) {
            int destination;
            if (bitArray[i] == 0) {
                destination = prefixSum[i];
            } else {
                destination = i - prefixSum[i] + totalFalses;
            }
            output[destination] = arr[i];
        }

        // Copy back to input array for next iteration
        std::copy(output.begin(), output.end(), arr.begin());
    }
}

void filtro_mediana_cpu(int* img_in, int* img_out, int width, int height, int W) {
    for (int pixel = 0; pixel < width * height; pixel++) {
        int x = pixel % width;
        int y = pixel / width;
        std::vector<int> window;

        for (int i = x - W; i <= x + W; i++) {
            for (int j = y - W; j <= y + W; j++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    window.push_back(img_in[j * width + i]);
                }
            }
        }

        radixSort_cpu(window);
        img_out[pixel] = window[window.size() / 2];
    }
}


 // ...................................................................................................................

#define BITS_PER_PASS 1

// Error checking macro
#define cudaCheckError() {                                      \
    cudaError_t e = cudaGetLastError();                         \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                \
    }                                                           \
}

//--------------------

#define BLOCK_SIZE_RADIX 256


// Función para realizar un split basado en el bit n-ésimo
__device__ void split(int* d_input, int* d_output, int* d_bitArray, int* d_prefixSum, int n, int size) {
    int idx = threadIdx.x;
    if (idx < size) {
        // Extract bit
        int as_int = d_input[idx];
        d_bitArray[idx] = (as_int >> n) & 1;
    }
    __syncthreads();

    // Perform exclusive scan (simplified for shared memory usage)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (idx + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            d_prefixSum[index] += d_prefixSum[index - stride];
        }
        __syncthreads();
    }

    if (idx < size) {
        // Reorder
        int destination;
        if (d_bitArray[idx] == 0) {
            destination = d_prefixSum[idx];
        } else {
            destination = idx - d_prefixSum[idx] + d_prefixSum[size-1] + (d_bitArray[size-1] == 0);
        }
        d_output[destination] = d_input[idx];
    }
    __syncthreads();

    // Copy back to input array for next iteration
    if (idx < size) {
        d_input[idx] = d_output[idx];
    }
}

// Función principal de Radix Sort
__device__ void radixSort(int* d_input, int size) {
    __shared__ int d_output[1024];  // Adjust size as needed, or use dynamic shared memory
    __shared__ int d_bitArray[1024];  // Adjust size as needed
    __shared__ int d_prefixSum[1024]; // Adjust size as needed

    for (int bit = 0; bit < 32; bit++) {
        split(d_input, d_output, d_bitArray, d_prefixSum, bit, size);
    }
}

// Kernel para aplicar el filtro de mediana
__global__ void medianaKernel(int* img_in, int* img_out, int width, int height, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel = y * width + x;

    if (x < width && y < height) {
        __shared__ int window[1024];  // Adjust size as needed, or use dynamic shared memory
        int count = 0;

        for (int i = x - W; i <= x + W; i++) {
            for (int j = y - W; j <= y + W; j++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    window[count++] = img_in[j * width + i];
                }
            }
        }

        radixSort(window, count);

        img_out[pixel] = window[count / 2];
    }
}

void filtro_mediana_gpu(int* img_in, int* img_out, int width, int height, int W) {
    int *d_img_in, *d_img_out;
    size_t size = width * height * sizeof(int);

    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);

    cudaMemcpy(d_img_in, img_in, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t sharedMemSize = (2 * W + 1) * (2 * W + 1) * sizeof(int);
    medianaKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_img_in, d_img_out, width, height, W);
    cudaDeviceSynchronize();
    cudaMemcpy(img_out, d_img_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img_in);
    cudaFree(d_img_out);
}


