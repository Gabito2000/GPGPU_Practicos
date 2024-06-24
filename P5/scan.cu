#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <chrono>

#define BASE_SIZE 1024 // Tamaño base del vector

// Implementación usando Thrust
void thrust_suma_exclusiva(int* d_in, int* d_out, int n) {
    thrust::device_ptr<int> dev_in(d_in);
    thrust::device_ptr<int> dev_out(d_out);
    // exclusive_scan: Esta función realiza un escaneo exclusivo en un arreglo de enteros en la GPU utilizando la biblioteca Thrust
    thrust::exclusive_scan(dev_in, dev_in + n, dev_out);
}

// Implementaciòn usando CUB
void cub_suma_exclusiva(int* d_in, int* d_out, int n) {
    // reservo las variables temporales que utilizará CUB
    void* d_temp = NULL;
    size_t temp_bytes = 0;

    // determino el tamaño necesario de la memoria temporal y lo almacena en temp_storage_bytes
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, n);
    cudaMalloc(&d_temp, temp_bytes);

    // Ahora, con d_temp_storage apuntando a la memoria temporal asignada, se llama nuevamente a cub::DeviceScan::ExclusiveSum.
    // Esta llamada realiza el ExclusiveSum, utilizando la memoria temporal para sus cálculos internos. Los resultados del escaneo se almacenan en el arreglo apuntado por d_out.
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, n);

    cudaFree(d_temp);
}


/* 
Implementación propia
    se ejecutara un bucle que ejecutará los siguientes pasos:
    1. sumará el máximo del bloque anterior a todos los elementos del bloque actual
    2. secuencialmente sumará todos los elementos anteriores del bloque actual a el elemento actual
*/
__global__ void propio_suma_exclusiva(int* d_in, int* d_out, int n) {
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    for (int i = 0; i < n; i += block_size) {
        int block_sum = 0;
        if (i > 0) {
            block_sum = d_out[i - 1];
        }
        
        // Suma el máximo del bloque anterior a todos los elementos del bloque actual
        if (i + tid < n) {
            d_out[i + tid] = block_sum;
        }
        __syncthreads();
        
        // Secuencialmente suma todos los elementos anteriores del bloque actual a el elemento actual
        for (int j = 0; j < block_size; ++j) {
            if (i + tid < n && j <= tid) {
                d_out[i + tid] += d_in[i + j];
            }
            __syncthreads();
        }
    }
}

void write_results_to_file(const char* filename, int* data, int n) {
    std::ofstream file(filename);
    for (int i = 0; i < n; ++i) {
        file << data[i] << "\n";
    }
    file.close();
}

int main() {
    int K = 3; // Cambia el valor de K según sea necesario
    int N = BASE_SIZE * (1 << K);

    int *h_in = new int[N];
    int *h_out = new int[N];
    int *d_in, *d_out;

    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Medir el tiempo de la implementación propia
    auto start = std::chrono::high_resolution_clock::now();
    propio_suma_exclusiva<<<1, 1024>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Tiempo de la implementación propia: " << diff.count() << " s\n";

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    write_results_to_file("resultados_propios.txt", h_out, N);

    // Medir el tiempo de la implementación con CUB
    start = std::chrono::high_resolution_clock::now();
    cub_suma_exclusiva(d_in, d_out, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Tiempo de la implementación con CUB: " << diff.count() << " s\n";

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    write_results_to_file("resultados_cub.txt", h_out, N);

    // Medir el tiempo de la implementación con Thrust
    start = std::chrono::high_resolution_clock::now();
    thrust_suma_exclusiva(d_in, d_out, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Tiempo de la implementación con Thrust: " << diff.count() << " s\n";

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    write_results_to_file("resultados_thrust.txt", h_out, N);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
