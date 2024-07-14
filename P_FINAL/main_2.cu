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
    
int main(int argc, char** argv){
	
	const char * path;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		path = argv[argc-1];

	CImg<int> image(path);
	CImg<int> image_out(image.width(), image.height(),1,1,0);

	int *img_matrix = image.data();
   int *img_out_matrix = image_out.data();

	int w = 3;
	std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++){
		filtro_mediana_cpu(img_matrix, img_out_matrix, image.width(), image.height(), w);
	}
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
	printf("version 2 ----------------------- \n");
	printf("Radixsort  \n");
	printf("Tiempo CPU: %f\n", duration.count()/ITERATIONS);
	image_out.save("output_cpu_2.pgm");
	
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < ITERATIONS; i++){
		filtro_mediana_gpu(img_matrix, img_out_matrix, image.width(), image.height(), w);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
	printf("Tiempo GPU: %f\n", duration.count()/ITERATIONS);
	image_out.save("output_gpu_2.pgm");
   	
   return 0;
}

