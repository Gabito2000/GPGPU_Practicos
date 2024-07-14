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
#include <thrust/iterator/constant_iterator.h>

using namespace std;

// Kernel to extract window elements and convert to integers
__global__ void extractWindowKernel(int* img_in, int* windows, int width, int height, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel = y * width + x;
    int windowSize = (2 * W + 1) * (2 * W + 1);
    if (x < width && y < height) {
        int count = 0;
        for (int j = y - W; j <= y + W; j++) {
            for (int i = x - W; i <= x + W; i++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    windows[pixel * windowSize + count] = img_in[j * width + i];
                }
                else {
                    windows[pixel * windowSize + count] = 0;  // Pad with 0 for out-of-bounds
                }
                count++;
            }
        }
    }
}

// Kernel to select median
__global__ void selectMedianKernel(int* sortedWindows, int* img_out, int width, int height, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel = y * width + x;
    int windowSize = (2 * W + 1) * (2 * W + 1);
    if (x < width && y < height) {
        img_out[pixel] = sortedWindows[pixel * windowSize + windowSize / 2];
    }
}

struct SortWindow {
    int* d_windows;
    int windowSize;
    
    SortWindow(int* _d_windows, int _windowSize) 
        : d_windows(_d_windows), windowSize(_windowSize) {}
    
    __device__
    void operator()(int i) const {
        thrust::sort(thrust::seq, d_windows + i * windowSize, d_windows + (i + 1) * windowSize);
    }
};

void filtro_mediana_gpu(int* img_in, int* img_out, int width, int height, int W) {
    int *d_windows;
    int *d_img_in, *d_img_out;
    size_t size = width * height * sizeof(int);
    int windowSize = (2 * W + 1) * (2 * W + 1);
    size_t windowsSize = width * height * windowSize * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);
    cudaMalloc(&d_windows, windowsSize);

    // Copy input image to device
    cudaMemcpy(d_img_in, img_in, size, cudaMemcpyHostToDevice);

    // Extract windows
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    extractWindowKernel<<<blocksPerGrid, threadsPerBlock>>>(d_img_in, d_windows, width, height, W);

    // Sort windows with thrust in parallel
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(width * height);
    thrust::for_each(thrust::device, begin, end, SortWindow(d_windows, windowSize));

    // Select median
    selectMedianKernel<<<blocksPerGrid, threadsPerBlock>>>(d_windows, d_img_out, width, height, W);

    // Copy output image to host
    cudaMemcpy(img_out, d_img_out, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_windows);
}