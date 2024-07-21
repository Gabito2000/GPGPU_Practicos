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

void radixSort_cpu(std::vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return;

    std::vector<int> output(n);
    std::vector<int> bitArray(n);
    std::vector<int> prefixSum(n);

    for (int bit = 0; bit < MAX_DIGITS; bit++) {
        int mask = 1 << bit;

        // Extracción de bit
        for (int i = 0; i < n; i++) {
            bitArray[i] = (arr[i] & mask) >> bit;
        }

        // Exclusive Scan
        prefixSum[0] = 0;
        for (int i = 1; i < n; i++) {
            prefixSum[i] = prefixSum[i - 1] + (1 - bitArray[i - 1]);
        }

        int totalFalses = prefixSum[n - 1] + (1 - bitArray[n - 1]);

        // Reordenamos
        std::fill(output.begin(), output.end(), 0); // LLenamos de 0s para evitar errores

        for (int i = 0; i < n; i++) {
            int destination;
            if (bitArray[i] == 0) {
                destination = prefixSum[i];
            } else {
                destination = i - prefixSum[i] + totalFalses;
            }
            output[destination] = arr[i];
        }

        // Copiamos de vuelta a arr
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
__global__ void fillWindows(int* img_in, int* windows, int width, int height, int W) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int pixel = x + y * width;

    if (pixel >= width * height) return;

    int windowSize = (2 * W + 1) * (2 * W + 1);
    int elemento_ventana = threadIdx.x + threadIdx.y * blockDim.x;

    if (elemento_ventana >= windowSize) return;

    int poss_inicio_windows_arr = pixel * windowSize; 
    int poss_elemento_en_ventana = elemento_ventana + poss_inicio_windows_arr;

    int img_x = x + (threadIdx.x - W);
    int img_y = y + (threadIdx.y - W);
    int elemento_de_la_ventana_que_copiaremos = img_y * width + img_x;

    if (img_x < 0 || img_x >= width || img_y < 0 || img_y >= height) {
        windows[poss_elemento_en_ventana] = 0;
    } else {
        windows[poss_elemento_en_ventana] = img_in[elemento_de_la_ventana_que_copiaremos];
    }

    __syncthreads();
}

__global__ void radixSort_gpu(int* windows, int width, int height, int W) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    if (x >= width || y >= height) return;
    
    int windowSize = (2 * W + 1) * (2 * W + 1);
    int tdx = threadIdx.x + threadIdx.y * blockDim.x;
    if (tdx >= windowSize) return;

    int pixel = x + y * width;
    int* currentWindow = &windows[pixel * windowSize];
    
    extern __shared__ int sharedMem[];
    int* output = sharedMem;
    int* bitArray = &sharedMem[windowSize];
    int* prefixSum = &sharedMem[2 * windowSize];

    __shared__ int totalFalses;

    for (int bit = 0; bit < MAX_DIGITS; bit++) {
        int mask = 1 << bit;

        // Extracción de bit
        bitArray[tdx] = (currentWindow[tdx] & mask) >> bit;
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
        output[destination] = currentWindow[tdx];
        __syncthreads();

        // Copiamos de vuelta a currentWindow
        currentWindow[tdx] = output[tdx];
        __syncthreads();
    }
}


__global__ void selectMedian(int* windows, int* img_out, int width, int height, int W) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * width;
    int windowSize = (2 * W + 1) * (2 * W + 1);
    if (pixel >= width * height) return;
    img_out[pixel] = windows[pixel * windowSize + windowSize / 2];
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
    fillWindows<<<blocksPerGrid, threadsPerBlock>>>(d_img_in, d_windows, width, height, W);
    cudaDeviceSynchronize();

    // Ordenamos las ventanas
    size_t sharedMemSize = 3 * windowSize * sizeof(int); // para output, bitArray y prefixSum
    radixSort_gpu<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_windows, width, height, W);
    cudaDeviceSynchronize();

    // Seleccionamos el elemento del medio
    dim3 threadsPerBlock2(32, 32);
    dim3 blocksPerGrid2((width + threadsPerBlock2.x - 1) / threadsPerBlock2.x, 
                        (height + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
    selectMedian<<<blocksPerGrid2, threadsPerBlock2>>>(d_windows, d_img_out, width, height, W);
    cudaDeviceSynchronize();

    cudaMemcpy(img_out, d_img_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_windows);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}


