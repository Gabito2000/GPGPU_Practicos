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

#define MAX_INT 2147483647
// Estructura para extraer la ventana alrededor de cada píxel
struct ExtractWindow {
    int* d_img_in;
    int* d_windows;
    int width;
    int height;
    int W;
    int windowSize;
    
    ExtractWindow(int* _d_img_in, int* _d_windows, int _width, int _height, int _W) 
        : d_img_in(_d_img_in), d_windows(_d_windows), width(_width), height(_height), W(_W) {
        windowSize = (2 * W + 1) * (2 * W + 1);
    }
    
    __device__
    void operator()(int i) const {
        int x = i % width;
        int y = i / width;
        int pixel = y * width + x;
        int count = 0;
        int padding = 0;
        for (int j = y - W; j <= y + W; j++) {
            for (int i = x - W; i <= x + W; i++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    d_windows[pixel * windowSize + count] = d_img_in[j * width + i];
                }
                else {
                    d_windows[pixel * windowSize + count] = padding;
                    padding = MAX_INT-padding; // Para evitar el padding
                }
                count++;
            }
        }
    }
};

// Estructura para seleccionar la mediana de cada ventana
struct SelectMedian {
    int* d_windows;
    int* d_img_out;
    int width;
    int height;
    int W;
    int windowSize;
    
    SelectMedian(int* _d_windows, int* _d_img_out, int _width, int _height, int _W) 
        : d_windows(_d_windows), d_img_out(_d_img_out), width(_width), height(_height), W(_W) {
        windowSize = (2 * W + 1) * (2 * W + 1);
    }
    
    __device__
    void operator()(int i) const {
        int x = i % width;
        int y = i / width;
        int pixel = y * width + x;
        d_img_out[pixel] = d_windows[pixel * windowSize + windowSize / 2];
    }
};


// Estructura para ordenar cada ventana utilizando Thrust
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

// Función para aplicar el filtro de mediana en la GPU
void filtro_mediana_gpu(int* img_in, int* img_out, int width, int height, int W) {
    int *d_windows;
    int *d_img_in, *d_img_out;
    size_t size = width * height * sizeof(int);
    int windowSize = (2 * W + 1) * (2 * W + 1);
    size_t windowsSize = width * height * windowSize * sizeof(int);

    // Asigno memoria 
    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);
    cudaMalloc(&d_windows, windowsSize);

    // Copio la imagen de entrada al dispositivo
    cudaMemcpy(d_img_in, img_in, size, cudaMemcpyHostToDevice);

    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(width * height);
    
    thrust::for_each(thrust::device, begin, end, ExtractWindow(d_img_in, d_windows, width, height, W));

    // Ordeno cada ventana
    thrust::for_each(thrust::device, begin, end, SortWindow(d_windows, windowSize));

    // selecciono mediana de cada ventana
    thrust::for_each(thrust::device, begin, end, SelectMedian(d_windows, d_img_out, width, height, W));

    // copio al host
    cudaMemcpy(img_out, d_img_out, size, cudaMemcpyDeviceToHost);

    //libero la memoria
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_windows);
}