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

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
using namespace std;



void filtro_mediana_cpu(int* img_in, int* img_out, int width, int height, int W) {
    //RETURN  THE INPUT
    for (int i = 0; i < width * height; i++) {
        img_out[i] = img_in[i];
    }
}

 // ...................................................................................................................

#define BITS_PER_PASS 1
#define MAX_DIGITS 32 // Assuming 32-bit integers

// Error checking macro
#define cudaCheckError() {                                      \
    cudaError_t e = cudaGetLastError();                         \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                \
    }                                                           \
}

// Kernel to extract window elements and convert to integers
__global__ void extractWindowKernel(int* img_in, int* windows, int width, int height, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel = y * width + x;
    int windowSize = (2 * W + 1) * (2 * W + 1);

    if (x < width && y < height) {
        int count = 0;
        for (int i = x - W; i <= x + W; i++) {
            for (int j = y - W; j <= y + W; j++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    windows[pixel * windowSize + count] = img_in[j * width + i];
                    count++;
                }
            }
        }
        // Pad the rest of the window with 0
        for (; count < windowSize; count++) {
            windows[pixel * windowSize + count] = 0;  // Max positive int
        }
    }
}

// Kernel to select median and convert back to int
__global__ void selectMedianKernel(int* sortedWindows, int* img_out, int width, int height, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel = y * width + x;
    int windowSize = (2 * W + 1) * (2 * W + 1);

    if (x < width && y < height) {
        img_out[pixel] = sortedWindows[pixel * windowSize + windowSize / 2];
    }
}

void filtro_mediana_gpu(int* img_in, int* img_out, int width, int height, int W) {
    int *d_windows, *d_sortedWindows;
    int *d_img_in, *d_img_out;
    size_t size = width * height * sizeof(int);
    int windowSize = (2 * W + 1) * (2 * W + 1);
    size_t windowsSize = width * height * windowSize * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);
    cudaMalloc(&d_windows, windowsSize);
    cudaMalloc(&d_sortedWindows, windowsSize);

    // Copy input image to device
    cudaMemcpy(d_img_in, img_in, size, cudaMemcpyHostToDevice);

    // Extract windows
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    extractWindowKernel<<<blocksPerGrid, threadsPerBlock>>>(d_img_in, d_windows, width, height, W);

    // Sort windows using CUB's DeviceRadixSort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // First, get the required temporary storage size
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_windows, d_sortedWindows, width * height * windowSize);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform the sort
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_windows, d_sortedWindows, width * height * windowSize);

    // Select median
    selectMedianKernel<<<blocksPerGrid, threadsPerBlock>>>(d_sortedWindows, d_img_out, width, height, W);

    // Copy result back to host
    cudaMemcpy(img_out, d_img_out, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_windows);
    cudaFree(d_sortedWindows);
    cudaFree(d_temp_storage);
}
