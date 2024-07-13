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

int getMax_cpu(const std::vector<int>& arr) {
    int max = arr[0];
    for (int num : arr) {
        if (num > max) {
            max = num;
        }
    }
    return max;
}

void countingSort_cpu(std::vector<int>& arr, int exp) {
    int n = arr.size();
    std::vector<int> output(n);
    int count[10] = {0};

    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

void radixSort_cpu(std::vector<int>& arr) {
    int max = getMax_cpu(arr);

    for (int exp = 1; max / exp > 0; exp *= 10) {
        countingSort_cpu(arr, exp);
    }
}

void filtro_mediana_cpu(float* img_in, float* img_out, int width, int height, int W) {
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        for (int pixel = 0; pixel < width * height; pixel++) {
            int x = pixel % width;
            int y = pixel / width;
            float window[(2 * W + 1) * (2 * W + 1)];
            int count = 0;
            for (int i = x - W; i <= x + W; i++) {
                for (int j = y - W; j <= y + W; j++) {
                    if (i >= 0 && i < width && j >= 0 && j < height) {
                        window[count++] = img_in[j * width + i];
                    }
                }
            }

            // Escalar a enteros
            std::vector<int> int_window(count);
            for (int k = 0; k < count; k++) {
                int_window[k] = static_cast<int>(window[k] * 1000); // Ajustar escala si es necesario
            }

            // Aplicar Radix Sort
            radixSort_cpu(int_window);

            // Desescalar a flotantes y encontrar la mediana
            for (int k = 0; k < count; k++) {
                window[k] = static_cast<float>(int_window[k]) / 1000.0f; // Ajustar escala si es necesario
            }

            img_out[pixel] = window[count / 2];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    printf("Tiempo CPU: %f\n", duration.count());
}

 // ...................................................................................................................

__device__ void radixSort(float* window, int count) {
    extern __shared__ int int_window[];
    for (int i = 0; i < count; i++) {
        int_window[i] = static_cast<int>(window[i] * 1000);
    }

    int max = int_window[0];
    for (int i = 1; i < count; i++) {
        if (int_window[i] > max) {
            max = int_window[i];
        }
    }

    for (int exp = 1; max / exp > 0; exp *= 10) {
        __shared__ int output[1024];
        __shared__ int countArr[10];

        if (threadIdx.x < 10) {
            countArr[threadIdx.x] = 0;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            atomicAdd(&countArr[(int_window[i] / exp) % 10], 1);
        }
        __syncthreads();

        for (int i = 1; i < 10; i++) {
            countArr[i] += countArr[i - 1];
        }
        __syncthreads();

        for (int i = count - 1; i >= 0; i--) {
            int idx = atomicSub(&countArr[(int_window[i] / exp) % 10], 1) - 1;
            output[idx] = int_window[i];
        }
        __syncthreads();

        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            int_window[i] = output[i];
        }
        __syncthreads();
    }

    for (int i = 0; i < count; i++) {
        window[i] = static_cast<float>(int_window[i]) / 1000.0f;
    }
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

        radixSort(window, count);

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