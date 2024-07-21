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

// Intercambiar dos elementos sin usar una variable temporal
using namespace std;
__device__ void swap(int& a, int& b) {
    a = a + b;
    b = a - b;
    a = a - b;
}

// Ordenar un arreglo usando bubble sort
__device__ void sort(int* window, int windowSize){
    for (int i = 0; i < windowSize; i++){
        for (int j = 0; j < windowSize - i - 1; j++){
            if (window[j] > window[j + 1]){
                swap(window[j], window[j + 1]);
            }
        }
    }
}

// Kernel para el filtro de mediana
__global__ void filtro_mediana_kernel(int* d_input, int* d_output, int width, int height, int W){

    int windowSize = (2 * W + 1) * (2 * W + 1);
    int* window = new int[windowSize];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height){
        int count = 0;
        for (int i = x - W; i <= x + W; i++){
            for (int j = y - W; j <= y + W; j++){
                if (i >= 0 && i < width && j >= 0 && j < height){
                    window[count++] = d_input[j * width + i];
                }
            }
        }

        sort(window, windowSize);
       
        d_output[y * width + x] = window[windowSize / 2];
    }
    delete[] window;
}


// funcion perincipal llamada desde el main, trabaja utilizando la gpu
void filtro_mediana_gpu(int * img_in, int * img_out, int width, int height, int W){
    int *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(int));
    cudaMalloc(&d_output, width * height * sizeof(int));

    cudaMemcpy(d_input, img_in, width * height * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize((W * 2 + 1), (W * 2 + 1));
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    filtro_mediana_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, W);

    cudaMemcpy(img_out, d_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);    
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

}

// implementacion que trabaja en CPU
void filtro_mediana_cpu(int * img_in, int * img_out, int width, int height, int W){
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
        // Ordeno el arreglo con quicksort
        std::sort(window, window + count);
        img_out[pixel] = window[count / 2];
    }
}
