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

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <thrust/iterator/constant_iterator.h>
#define MAX_DIGITS 32 // Asumiendo 32-bit integers
#define MAX_INT 3000

using namespace std;

__global__ void gpuUnificado(int* img_in, int* windows, int width, int height, int W, int* img_out) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pixel = x + y * width;

    int windowSize = (2 * W + 1) * (2 * W + 1);
    extern __shared__ int sharedMem[];
    int* output = sharedMem;
    int* bitArray = &sharedMem[windowSize];
    int* prefixSum = &sharedMem[2 * windowSize];
    int* currentWindowCopy = &sharedMem[3 * windowSize];


    if (pixel >= width * height) return;

    int elemento_ventana = threadIdx.x + threadIdx.y * blockDim.x;

    if (elemento_ventana >= windowSize) return;


    int img_x = x + (threadIdx.x - W);
    int img_y = y + (threadIdx.y - W);
    int elemento_de_la_ventana_que_copiaremos = img_y * width + img_x;

    if (img_x < 0 || img_x >= width || img_y < 0 || img_y >= height) {
        currentWindowCopy[elemento_ventana] = 0;
    } else {
        currentWindowCopy[elemento_ventana] = img_in[elemento_de_la_ventana_que_copiaremos];
    }

    __syncthreads();

    // Ordenamos las ventanas
    if (x >= width || y >= height) return;
    
    int tdx = threadIdx.x + threadIdx.y * blockDim.x;
    if (tdx >= windowSize) return;
    
    __shared__ int totalFalses;

    for (int bit = 0; bit < MAX_DIGITS; bit++) {
        int mask = 1 << bit;

        // Extracción de bit
        bitArray[tdx] = (currentWindowCopy[tdx] & mask) >> bit;
        __syncthreads();

        // Exclusive Scan
        if (tdx == 0) {
            prefixSum[0] = 0;
            for (int i = 1; i < windowSize; i++) {
                prefixSum[i] = prefixSum[i-1] + (1 - bitArray[i - 1]);
            }
            totalFalses = prefixSum[windowSize - 1] + (1 - bitArray[windowSize - 1]);
        }
        __syncthreads();
    
        // Reordenamos
        int destination;
        if (bitArray[tdx] == 0) {
            destination = prefixSum[tdx];
        } else {
            destination = tdx - prefixSum[tdx] + totalFalses;
        }
        output[destination] = currentWindowCopy[tdx];
        __syncthreads();

        // Copiamos de vuelta a currentWindow
        currentWindowCopy[tdx] = output[tdx];
        __syncthreads();
    }

    // Seleccionamos el elemento del medio
    img_out[pixel] = currentWindowCopy[windowSize / 2];

}

void filtro_mediana_gpu(int* img_in, int* img_out, int width, int height, int W) {
    int *d_img_in, *d_img_out, *d_windows;
    size_t size = width * height * sizeof(int);
    int windowSize = (2 * W + 1) * (2 * W + 1);

    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);
    cudaMalloc(&d_windows, width * height * windowSize * sizeof(int));

    cudaMemcpy(d_img_in, img_in, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2 * W + 1, 2 * W + 1);
    dim3 blocksPerGrid(width, height);

    // LLenar ventana
    size_t sharedMemSize = 4 * windowSize * sizeof(int); // para output, bitArray y prefixSum
    gpuUnificado<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_img_in, d_windows, width, height, W, d_img_out);

    cudaMemcpy(img_out, d_img_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_windows);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}


