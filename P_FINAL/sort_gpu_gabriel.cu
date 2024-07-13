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

using namespace std;

#define BLOCK_SIZE_RADIX 256

// Kernel for computing the exclusive scan (prefix sum)
__global__ void exclusiveScanKernel(int* input, int* output, int n) {
    __shared__ int temp[2 * BLOCK_SIZE_RADIX];
    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    int ai = tid;
    int bi = tid + BLOCK_SIZE_RADIX;
    temp[ai] = (ai < n) ? input[ai] : 0;
    temp[bi] = (bi < n) ? input[bi] : 0;

    // Build sum in place up the tree
    for (int d = BLOCK_SIZE_RADIX; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (tid == 0) {
        temp[2 * BLOCK_SIZE_RADIX - 1] = 0;
    }

    // Traverse down tree & build scan
    for (int d = 1; d < 2 * BLOCK_SIZE_RADIX; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write results to output array
    if (ai < n) output[ai] = temp[ai];
    if (bi < n) output[bi] = temp[bi];
}

// Kernel for performing the split operation
__global__ void splitKernel(int* input, int* output, int n, int bit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int value = input[tid];
        int b = (value >> bit) & 1;
        int f = 1 - b;
        
        __shared__ int totalFalses;
        if (threadIdx.x == 0) {
            totalFalses = 0;
        }
        __syncthreads();
        
        atomicAdd(&totalFalses, f);
        __syncthreads();
        
        int index = (b == 0) ? atomicAdd(&totalFalses, 0) : tid - totalFalses + n;
        output[index] = value;
    }
}

// Host function to perform Radix Sort
void radixSort(int* d_input, int* d_output, int n) {
    int* d_temp;
    cudaMalloc(&d_temp, n * sizeof(int));

    for (int bit = 0; bit < 32; ++bit) {
        splitKernel<<<(n + BLOCK_SIZE_RADIX - 1) / BLOCK_SIZE_RADIX, BLOCK_SIZE_RADIX>>>(d_input, d_output, n, bit);
        cudaDeviceSynchronize();

        // Swap input and output for next iteration
        int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    // Ensure the final sorted array is in d_output
    if (d_input != d_output) {
        cudaMemcpy(d_output, d_input, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_temp);
}

// Kernel para aplicar el filtro de mediana
__global__ void medianaKernel(float* img_in, float* img_out, int width, int height, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel = y * width + x;

    if (x < width && y < height) {
        extern __shared__ float window[];
        int count = 0;

        for (int i = x - W; i <= x + W; i++) {
            for (int j = y - W; j <= y + W; j++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    window[count++] = img_in[j * width + i];
                }
            }
        }

        radixSort(window, window, count);

        img_out[pixel] = window[count / 2];
    }
}

void filtro_mediana_gpu(float* img_in, float* img_out, int width, int height, int W) {
    float *d_img_in, *d_img_out;
    size_t size = width * height * sizeof(float);

    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);

    cudaMemcpy(d_img_in, img_in, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();

    size_t sharedMemSize = (2 * W + 1) * (2 * W + 1) * sizeof(float);
    for (int i = 0; i < 10; i++) {
        medianaKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_img_in, d_img_out, width, height, W);
        cudaDeviceSynchronize();
    }

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    printf("Tiempo GPU: %f\n", duration.count());

    cudaMemcpy(img_out, d_img_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img_in);
    cudaFree(d_img_out);
}