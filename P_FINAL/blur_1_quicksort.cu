#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvToolsExt.h"
#include <cuda_runtime.h>
#include <algorithm> 
#include <vector>
#include <iostream>


using namespace std;


__device__ void swap(int& a, int& b) {
    //swap two elements without using a temporary variable
    a = a + b;
    b = a - b;
    a = a - b;
}

__device__ int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

__device__ void quickSort(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

__global__ void filtro_mediana_kernel(int* d_input, int* d_output, int width, int height, int W){

    int windowSize = (2 * W + 1) * (2 * W + 1);
    int* window = new int[windowSize];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int count = 0;

    for (int i = x - W; i <= x + W; i++){
        for (int j = y - W; j <= y + W; j++){
            if (i >= 0 && i < width && j >= 0 && j < height){
                window[count++] = d_input[j * width + i];
            }
        }
    }

    //sort array with a quicksort
    quickSort(window, 0, count - 1);
    d_output[y * width + x] = window[count / 2];

}

void filtro_mediana_gpu(int * img_in, int * img_out, int width, int height, int W){
    int *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(int));
    cudaMalloc(&d_output, width * height * sizeof(int));

    cudaMemcpy(d_input, img_in, width * height * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    filtro_mediana_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, W);

    cudaMemcpy(img_out, d_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);    
}

void filtro_mediana_cpu(int * img_in, int * img_out, int width, int height, int W){
    //implementar filtro de mediana en CPU 
    for (int pixel = 0; pixel < width * height; pixel++){
        int x = pixel % width;
        int y = pixel / width;
        int window[(2 * W + 1) * (2 * W + 1)];
        int count = 0;
        for (int i = x - W; i <= x + W; i++){
            for (int j = y - W; j <= y + W; j++){
                if (i >= 0 && i < width && j >= 0 && j < height){
                    window[count++] = img_in[j * width + i];
                }
            }
        }
        //sort array with a quicksort
        std::sort(window, window + count);
        img_out[pixel] = window[count / 2];
    }
}
