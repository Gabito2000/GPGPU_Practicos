#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <iostream>
#include <string>

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
int get_text_length(const char * fname);

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

#define N 512

#define BLOCK_SIZE 256
#define NUM_BLOCKS 128


__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

// PARTE 1 A
/*
__global__ void decrypt_kernel(int *d_message, int length)
{
    int tid = threadIdx.x;

    for (int i = tid; i < length; i += blockDim.x) {
        d_message[i] = modulo(A_MMI_M * (d_message[i] - B), M);
    }
}*/



// PARTE 1 B
__global__ void decrypt_kernel(int *d_message, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // Total number of threads across all blocks
    
    for (int i = idx; i < length; i += stride) {
        d_message[i] = modulo(A_MMI_M * (d_message[i] - B), M);
    }
}

// PARTE 2
__global__ void count_occurrences_kernel(int *d_message, int length, int *d_occurrences)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < length; i += stride) {
        atomicAdd(&d_occurrences[d_message[i]], 1);
    }
}


void print_cuda_error(const std::string &message) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cout << message << " : " << cudaGetErrorString(error) << "\n";
		std::cout << "Error OUT \n";
		exit(1);
	}
}

int main(int argc, char *argv[])
{
	int *h_message;
	int *d_message;
	unsigned int size;
	const char * fname;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	/* reservar memoria en la GPU */
	CUDA_CHK(cudaMalloc((void**)&d_message, size));
	print_cuda_error("Error al reservar memoria en la GPU");
		
	/* copiar los datos de entrada a la GPU */
	CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));
	print_cuda_error("Error al copiar los datos de entrada a la GPU");

	/* Configurar la grilla y lanzar el kernel */
	//int blockSize = 256;
	
	/* Configurar la grilla y lanzar el kernel */
    dim3 blockSize(BLOCK_SIZE);
    dim3 numBlocks(NUM_BLOCKS);
	
	/* Copiar los datos de salida a la CPU en h_message */
	//decrypt_kernel<<<1, blockSize>>>(d_message, length);
	//decrypt_kernel<<<(length + blockSize - 1) / blockSize, blockSize>>>(d_message, length);
	decrypt_kernel<<<numBlocks, blockSize>>>(d_message, length);
    
	print_cuda_error("Error al lanzar el kernel");

	/* Retorno los datos para prinearlos*/
	CUDA_CHK(cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost));
	print_cuda_error("Error al copiar los datos de salida a la CPU");

	// despliego el mensaje
	for (int i = 0; i < length; i++) {
		printf("%c", (char) h_message[i]);
	}
	printf("\n");

	//save it to a file
	FILE *f = fopen("decrypted.txt", "w");
	for (int i = 0; i < length; i++) {
		fprintf(f, "%c", (char) h_message[i]);
	}


	// libero la memoria en la GPU
	CUDA_CHK(cudaFree(d_message));

	// libero la memoria en la CPU
	free(h_message);
}

	
int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);    
	fseek(f, 0, SEEK_END);    
	size_t length = ftell(f); 
	fseek(f, pos, SEEK_SET);  

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c; 
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}