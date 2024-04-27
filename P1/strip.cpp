#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <algorithm> // for std::min
#include <malloc.h>


typedef float VALT;

void initMatrix(VALT* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10; // Generate random numbers between 0 and 9
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

double multiplicationViejo(int m, int n, int p, VALT * A, VALT * B, VALT * C) {
    clock_t start = clock();
    for (int row = 0; row < m; row++) {
        for (int it = 0; it < p; it++) {
            for (int col = 0; col < n; col++) {
                C[row * n + col] += A[row * p + it] * B[it * n + col];
            }
        }
    }
    clock_t end = clock();
    double elapsed_seconds = double(end - start) / CLOCKS_PER_SEC;
    double flops = 2.0 * m * n * p;
    double gflops = (flops / elapsed_seconds) / 1e9; 
    return gflops;
}

double multiplicationOptimizado(int m, int n, int p, VALT* A, VALT* B, VALT* C, int strip_size) {
    clock_t start = clock();
    int col_end = 0;
    for (int row = 0; row < m; row++) {
        for(int it = 0; it < p; it++) {
            for (int col = 0; col < n; col += strip_size) {
                col_end = std::min(col + strip_size, n);
                for (int j = col; j < col_end; j++) {
                    C[row * n + j] += A[row * p + it] * B[it * n + j];
                }
            }
        }
    }
    clock_t end = clock();
    double elapsed_seconds = double(end - start) / CLOCKS_PER_SEC;
    double flops = 2.0 * m * n * p;
    double gflops = (flops / elapsed_seconds) / 1e9;
    return gflops;
}

double executeExperiment(int strip_size, int iteraciones) {
    double gfviejo = 0;
    double gfnuevo = 0;

    for (int i = 0; i < iteraciones; i++) {
        int m = 300;
        int n = 300;
        int p = 150;

        VALT* A = (VALT*)aligned_alloc(64, m * p * sizeof(VALT));
        VALT* B = (VALT*)aligned_alloc(64, p * n * sizeof(VALT));
        VALT* C = (VALT*)aligned_alloc(64, m * n * sizeof(VALT));
        VALT* C2 = (VALT*)aligned_alloc(64, m * n * sizeof(VALT));


        initMatrix(A, m, p);
        initMatrix(B, p, n);

        gfviejo += multiplicationViejo(m, n, p, A, B, C);

        gfnuevo += multiplicationOptimizado(m, n, p, A, B, C2, strip_size); 


        free(A);
        free(B);
        free(C);
        free(C2);
    }

    // std::cout << "gfnuevo " << gfnuevo << std::endl;
    // std::cout << "gfviejo " << gfviejo << std::endl;

    return (gfnuevo - gfviejo)/iteraciones;
}
void resetResults() {
    // Reset the results file to 'x, y\n'
    FILE *f = fopen("results.csv", "w");
    if (f == NULL) {
        std::cout << "Error opening file!" << std::endl;
        exit(1);
    }
    fprintf(f, "x, y\n");
    fclose(f);
}

void saveResults(double encontrado, int strip_size) {
    FILE *f = fopen("results.csv", "a");
    if (f == NULL) {
        std::cout << "Error opening file!" << std::endl;
        exit(1);
    }
    fprintf(f, "%d, %f\n", strip_size, encontrado);
    fclose(f);
}

int main() {
    srand(100);
    resetResults ();
    int iteraciones = 1;
    std::vector<int> mejores;

    int max = 0;
    double max_gflops = 0;

    int min = 0;
    double min_gflops = 0;
    
    for (int strip_size = 1; strip_size < 100; strip_size++) {
        double encontrado = executeExperiment(strip_size, iteraciones);


        // save the encontrado to resuts.csv file
        saveResults(encontrado, strip_size);
        if (encontrado > 0) {
            mejores.push_back(strip_size);

            if (encontrado > max_gflops) {
                max_gflops = encontrado;
                max = strip_size;
                std::cout << "Found a better one: " << strip_size << " with " << encontrado << " GFLOPS of diff" << std::endl;
            }
        }else {
            if (encontrado < min_gflops) {
                min_gflops = encontrado;
                min = strip_size;
                std::cout << "Found a worst one: " << strip_size << " with " << encontrado << " GFLOPS of diff" << std::endl;
            }
        }

        if (strip_size % 100 == 0)
            std::cout << "strip_size: " << strip_size << std::endl;
    }
    std::cout << "The best strip_size are: ";
    for (int i = 0; i < mejores.size(); i++) {
        std::cout << mejores[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "The best strip_size is: " << max << " with " << max_gflops << " GFLOPS of diff" << std::endl;
    std::cout << "The worst strip_size is: " << min << " with " << min_gflops << " GFLOPS of diff" << std::endl;

    std::cout << "Executed" << std::endl;
    return 0;
}
