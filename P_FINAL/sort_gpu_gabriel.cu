#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>
#include "CImg.h"
#define ITERATIONS 10

using namespace cimg_library;

// Function declarations
void filtro_mediana_gpu(int* img_in, int* img_out, int width, int height, int W);
void filtro_mediana_cpu(int* img_in, int* img_out, int width, int height, int W);

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <image_path> <version>\n", argv[0]);
        return -1;
    }

    const char* path = argv[1];
    int version = atoi(argv[2]);

    CImg<int> image(path);
    CImg<int> image_out(image.width(), image.height(), 1, 1, 0);

    int* img_matrix = image.data();
    int* img_out_matrix = image_out.data();

    int w = 3;
    std::chrono::high_resolution_clock::time_point start, end;

    // CPU timing
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        filtro_mediana_cpu(img_matrix, img_out_matrix, image.width(), image.height(), w);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    printf("Version %d\n", version);
    printf("Tiempo CPU: %f\n", duration.count() / ITERATIONS);
    char cpu_output_filename[50];
    snprintf(cpu_output_filename, sizeof(cpu_output_filename), "output_cpu_%d.pgm", version);
    image_out.save(cpu_output_filename);

    // GPU timing
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        filtro_mediana_gpu(img_matrix, img_out_matrix, image.width(), image.height(), w);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    printf("Tiempo GPU: %f\n", duration.count() / ITERATIONS);
    char gpu_output_filename[50];
    snprintf(gpu_output_filename, sizeof(gpu_output_filename), "output_gpu_%d.pgm", version);
    image_out.save(gpu_output_filename);

    return 0;
}
