#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "chrono"
#include "CImg.h"
#define ITERATIONS 10
using namespace cimg_library;

void filtro_mediana_gpu(int * img_in, int * img_out, int width, int height, int W);
void filtro_mediana_cpu(int * img_in, int * img_out, int width, int height, int W);

// Función para calcular la diferencia de tiempo en segundos
double time_diff(struct timeval *start, struct timeval *end) {
    return (double)(end->tv_sec - start->tv_sec) + (double)(end->tv_usec - start->tv_usec) / 1000000.0;
}

int main(int argc, char** argv) {
    const char *path;

    if (argc < 2) {
        printf("Debe ingresar el nombre del archivo\n");
        return 1;
    } else {
        path = argv[argc-1];
    }

    CImg<int> image(path);
    CImg<int> image_out(image.width(), image.height(), 1, 1, 0);

    int *img_matrix = image.data();
    int *img_out_matrix = image_out.data();

    int w[] = {1,2, 3, 5, 6}; 

    printf("version 4 ----------------------- \n");
    printf("Paralelismo \n");

    for (int j = 0; j < (sizeof(w) / sizeof(w[0])); j++) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (int i = 0; i < ITERATIONS; i++) {
            filtro_mediana_gpu(img_matrix, img_out_matrix, image.width(), image.height(), w[j]);
        }
        gettimeofday(&end, NULL);
        double duration = time_diff(&start, &end);
        printf("Tiempo GPU con (w=%d) : %f\n", w[j], duration / ITERATIONS);

        char nombreArchivo[50];
        sprintf(nombreArchivo, "output_gpu_4_w%d.pgm", w[j]);
        image_out.save(nombreArchivo);
    }

    return 0;
}