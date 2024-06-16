#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#define N 1024 // Tamaño del vector

// Función para usar Thrust para el escaneo exclusivo
void thrust_exclusive_scan(int* d_in, int* d_out, int n) {
    thrust::device_ptr<int> dev_in(d_in);
    thrust::device_ptr<int> dev_out(d_out);

    thrust::exclusive_scan(dev_in, dev_in + n, dev_out);
}


// Kernel para usar CUB para el escaneo exclusivo
void cub_exclusive_scan(int* d_in, int* d_out, int n) {
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);

    // Free temporary storage
    cudaFree(d_temp_storage);
}


__global__ void exclusive_scan_kernel(int* d_in, int* d_out, int n) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int offset = 1;

    int ai = tid;
    int bi = tid + blockDim.x;

    if (ai < n) s_data[ai] = d_in[ai];
    if (bi < n) s_data[bi] = d_in[bi];

    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            if (bi < n) s_data[bi] += s_data[ai];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        s_data[2 * blockDim.x - 1] = 0;
    }

    for (int d = 1; d < 2 * blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            if (bi < n) {
                int t = s_data[ai];
                s_data[ai] = s_data[bi];
                s_data[bi] += t;
            }
        }
    }
    __syncthreads();

    if (ai < n) d_out[ai] = s_data[ai];
    if (bi < n) d_out[bi] = s_data[bi];
}



int main() {
    int h_in[N], h_out[N];
    int *d_in, *d_out;

    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Medir el tiempo de la implementación propia
    auto start = std::chrono::high_resolution_clock::now();
    exclusive_scan_kernel<<<(N + 511) / 512, 512, 2 * 512 * sizeof(int)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Tiempo de la implementación propia: " << diff.count() << " s\n";

    // Medir el tiempo de la implementación con CUB
    start = std::chrono::high_resolution_clock::now();
    cub_exclusive_scan(d_in, d_out, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Tiempo de la implementación con CUB: " << diff.count() << " s\n";

    // Medir el tiempo de la implementación con Thrust
    start = std::chrono::high_resolution_clock::now();
    thrust_exclusive_scan(d_in, d_out, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Tiempo de la implementación con Thrust: " << diff.count() << " s\n";

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir resultados (opcional, para verificación)
    for (int i = 0; i < N; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}