#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef float VALT;

uint64_t get_nanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}

void initMatrix(VALT* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10; // Genera números aleatorios entre 0 y 9
    }
}

void printMatrix(VALT* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

double experimento1 (int experiments, VALT* A, VALT* B, VALT* C, int m, int n, int p) {
    double GFLOPS = 0;
    for (int c = 0; c < experiments; c++) {
        uint64_t start = get_nanoseconds();
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                for (int it = 0; it < p; it++) {
                    C[row * n + col] += A[row * p + it] * B[it * n + col];
                }
            }
        }
        uint64_t end = get_nanoseconds();

        // std::cout << "Time: " << end - start << " ns" << std::endl;
        // std::cout << "GFLOPS: " << (2.0 * m * n * p) / (end - start) << std::endl;
        // std::cout << "GFLOPS: " << (2.0 * m * n * p) / (end - start) << std::endl;
        GFLOPS += (2.0 * m * n * p) / (end - start); 
    }
    return GFLOPS / experiments;
}

double experimento2 (int experiments, VALT* A, VALT* B, VALT* C2, int m, int n, int p) {
    double GFLOPS = 0;
    for (int c = 0; c < experiments; c++) {

        for (int i = 0; i < m * n; i++) {
            C2[i] = 0;
        }

        VALT sum = 0;
        uint64_t start = get_nanoseconds();
        
        for (int row = 0; row < m; row++) {
            for (int it = 0; it < p; it++) {
                for (int col = 0; col < n; col++) {
                    C2[row * n + col] += A[row * p + it] * B[it * n + col];
                }
            }
        }


        uint64_t end = get_nanoseconds();

        GFLOPS += (2.0 * m * n * p) / (end - start); 
        
    }
    return GFLOPS / experiments;
}


int main() {
    int m = 300;
    int n = 300;
    int p = 100;
    
    int experiments = 1;
    
    // VALT A[m * p];
    // VALT B[p * n];
    // VALT C[m * n];

    VALT* A = (VALT*)_aligned_malloc(m * p * sizeof(VALT), 64);
    VALT* B = (VALT*)_aligned_malloc(p * n * sizeof(VALT), 64);
    VALT* C = (VALT*)_aligned_malloc(m * n * sizeof(VALT), 64);
    VALT* C2 = (VALT*)_aligned_malloc(m * n * sizeof(VALT), 64);


    initMatrix(A, m, p);
    initMatrix(B, p, n);

    std::cout << "GFLOPS1: " << experimento1(experiments, A, B, C, m, n, p) << std::endl;

    std::cout << "GFLOPS2: " << experimento2(experiments, A, B, C2, m, n, p) << std::endl;

    // free
    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C);
    _aligned_free(C2);

    return 0;
}
