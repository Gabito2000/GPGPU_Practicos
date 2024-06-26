#include "mmio.h"
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
#define WARP_PER_BLOCK 32
#define WARP_SIZE 32
#define CUDA_CHK(call) print_cuda_state(call);
#define MAX(A,B)        (((A)>(B))?(A):(B))
#define MIN(A,B)        (((A)<(B))?(A):(B))




static inline void print_cuda_state(cudaError_t code){

   if (code != cudaSuccess) printf("\ncuda error: %s\n", cudaGetErrorString(code));
   
}


__global__ void kernel_analysis_L(const int* __restrict__ row_ptr,
	const int* __restrict__ col_idx,
	volatile int* is_solved, int n,
	unsigned int* niveles) {
	extern volatile __shared__ int s_mem[];

	if(threadIdx.x==0&&blockIdx.x==0) printf("%i\n", WARP_PER_BLOCK);
	int* s_is_solved = (int*)&s_mem[0];
	int* s_info = (int*)&s_is_solved[WARP_PER_BLOCK];

	int wrp = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
	int local_warp_id = threadIdx.x / WARP_SIZE;

	int lne = threadIdx.x & 0x1f;                   // identifica el hilo dentro el warp

	if (wrp >= n) return;

	int row = row_ptr[wrp];
	int start_row = blockIdx.x * WARP_PER_BLOCK;
	int nxt_row = row_ptr[wrp + 1];

	int my_level = 0;
	if (lne == 0) {
		s_is_solved[local_warp_id] = 0;
		s_info[local_warp_id] = 0;
	}

	__syncthreads();

	int off = row + lne;
	int colidx = col_idx[off];
	int myvar = 0;

	while (off < nxt_row - 1)
	{
		colidx = col_idx[off];
		if (!myvar)
		{
			if (colidx > start_row) {
				myvar = s_is_solved[colidx - start_row];

				if (myvar) {
					my_level = max(my_level, s_info[colidx - start_row]);
				}
			} else
			{
				myvar = is_solved[colidx];

				if (myvar) {
					my_level = max(my_level, niveles[colidx]);
				}
			}
		}

		if (__all_sync(__activemask(), myvar)) {

			off += WARP_SIZE;
			//           colidx = col_idx[off];
			myvar = 0;
		}
	}
	__syncwarp();
	
	for (int i = 16; i >= 1; i /= 2) {
		my_level = max(my_level, __shfl_down_sync(__activemask(), my_level, i));
	}

	if (lne == 0) {

		s_info[local_warp_id] = 1 + my_level;
		s_is_solved[local_warp_id] = 1;
		niveles[wrp] = 1 + my_level;

		__threadfence();

		is_solved[wrp] = 1;
	}
}

    int* RowPtrL_d, *ColIdxL_d;
    VALUE_TYPE* Val_d;

// Definición del struct LevelSize
struct LevelSize {
    
    int level;
    int size;

     __host__ __device__
    bool operator==(const LevelSize& other) const {
        return (level == other.level && size == other.size);
    }
    __host__ __device__
    bool operator<(const LevelSize& other) const {
        return (level < other.level) || (level == other.level && size < other.size);
    }

};
struct LevelSizeTransform {
    __host__ __device__
    int operator()(const thrust::tuple<LevelSize, int>& t) const {
        LevelSize level_size_pair = thrust::get<0>(t);
        int count = thrust::get<1>(t);
        int size = (level_size_pair.size == 6) ? 1 : (1 << level_size_pair.size);
        return (count * size + WARP_SIZE - 1) / WARP_SIZE;
    }
};


// La estructura guarda un Puntero a los índices de las filas de la matriz y a los niveles.
struct func_LevelSize {
    int* row_ptr;
    unsigned int* niveles;

    func_LevelSize(int* _row_ptr, unsigned int* _niveles) : row_ptr(_row_ptr), niveles(_niveles) {}

    __host__ __device__
    LevelSize operator()(const int& idx) const {
        int level = niveles[idx];
        int nnz_row = row_ptr[idx + 1] - row_ptr[idx] - 1;
        int size;

        if (nnz_row == 0)
            size = 6;
        else if (nnz_row == 1)
            size = 0;
        else if (nnz_row <= 2)
            size = 1;
        else if (nnz_row <= 4)
            size = 2;
        else if (nnz_row <= 8)
            size = 3;
        else if (nnz_row <= 16)
            size = 4;
        else
            size = 5;

        return LevelSize{ level, size };
    }
};


int ordenar_filas(int* RowPtrL, int* ColIdxL, VALUE_TYPE* Val, int n, int* iorder,int veces) {
    // Crear vectores en dispositivo con Thrust
    thrust::device_vector<unsigned int> d_niveles(n);
    thrust::device_vector<int> d_is_solved(n);

    // Inicializar vectores en el dispositivo
    CUDA_CHK(cudaMemset(thrust::raw_pointer_cast(d_is_solved.data()), 0, n * sizeof(int)));
    CUDA_CHK(cudaMemset(thrust::raw_pointer_cast(d_niveles.data()), 0, n * sizeof(unsigned int)));

    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int grid = (n * WARP_SIZE + num_threads - 1) / num_threads;

    // Ejecutar kernel
    kernel_analysis_L<<<grid, num_threads, WARP_PER_BLOCK * (2 * sizeof(int))>>>(
        RowPtrL, 
        ColIdxL, 
        thrust::raw_pointer_cast(d_is_solved.data()),
        n, 
        thrust::raw_pointer_cast(d_niveles.data())
    );

    // Asegurarse de que el kernel se ejecutó correctamente
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Inicializar variables para medir el tiempo
    std::chrono::high_resolution_clock::time_point start, end;

    // Precrear vectores de Thrust reutilizables fuera del bucle
    thrust::device_vector<int> d_indices(n);
    thrust::device_vector<LevelSize> d_level_size_pairs(n);
    thrust::device_vector<LevelSize> d_level_1(n);
    thrust::device_vector<int> d_level_2(n);
    int n_warps;
    double resultados[veces];
    
    for(int i =0; i<veces;i++){ //todo borrar luego de la prueba
        start = std::chrono::high_resolution_clock::now();

        // Crear y llenar vector de índices
        thrust::sequence(d_indices.begin(), d_indices.end());

        // Transformar los índices a LevelSize
        thrust::transform(d_indices.begin(), d_indices.end(), d_level_size_pairs.begin(),
                          func_LevelSize(RowPtrL, thrust::raw_pointer_cast(d_niveles.data())));

        // Ordenar por nivel y tamaño
        thrust::stable_sort_by_key(d_level_size_pairs.begin(), d_level_size_pairs.end(), d_indices.begin());

        // Copiar resultados a iorder
        thrust::copy(d_indices.begin(), d_indices.end(), iorder);

        // Reducir por clave para contar el número de warps
        auto redu = thrust::reduce_by_key(d_level_size_pairs.begin(), d_level_size_pairs.end(),
                                          thrust::constant_iterator<int>(1), d_level_1.begin(), d_level_2.begin());

        // Calcular el número de clases
        int num_clases = std::distance(d_level_1.begin(), redu.first);

        // Transformar a warps
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(d_level_1.begin(), d_level_2.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(d_level_1.begin() + num_clases, d_level_2.begin() + num_clases)),
            d_level_2.begin(),
            LevelSizeTransform()
        );

        // Reducir para obtener el número total de warps
        n_warps = thrust::reduce(d_level_2.begin(), d_level_2.begin() + num_clases, 0);

        end = std::chrono::high_resolution_clock::now();
        resultados[i] = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    
    }
    // Calcular promedio y desviación estándar
    if(veces == 10){
        double promedio = 0;
        for(int i = 0; i < veces; i++) {
            promedio += resultados[i];
        }
        promedio /= veces;

        double desviacion = 0;
        for(int i = 0; i < veces; i++) {
            desviacion += (resultados[i] - promedio) * (resultados[i] - promedio);
        }
        desviacion = sqrt(desviacion / veces);

        printf("PromedioInterno: %f\n", promedio);
        printf("DesviaciónInterno: %f\n", desviacion);
    }

    return n_warps;
}






int main(int argc, char** argv)
{
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char* precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char*)"32-bit Single Precision";
    } else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char*)"64-bit Double Precision";
    } else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);


    int m, n, nnzA;
    int* csrRowPtrA;
    int* csrColIdxA;
    VALUE_TYPE* csrValA;

    int argi = 1;

    char* filename;
    if (argc > argi)
    {
        filename = argv[argi];
        argi++;
    }

    printf("-------------- %s --------------\n", filename);



    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE* f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_complex(matcode))
    {
        printf("Sorry, data type 'COMPLEX' is not supported.\n");
        return -3;
    }

    char* pch, * pch1;
    pch = strtok(filename, "/");
    while (pch != NULL) {
        pch1 = pch;
        pch = strtok(NULL, "/");
    }

    pch = strtok(pch1, ".");


    if (mm_is_pattern(matcode)) { isPattern = 1; }
    if (mm_is_real(matcode)) { isReal = 1;  }
    if (mm_is_integer(matcode)) { isInteger = 1; }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;


    if (n != m)
    {
        printf("Matrix is not square.\n");
        return -5;
    }

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric = 1;
        printf("input matrix is symmetric = true\n");
    } else
    {
        printf("input matrix is symmetric = false\n");
    }

    int* csrRowPtrA_counter = (int*)malloc((m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    int* csrRowIdxA_tmp = (int*)malloc(nnzA_mtx_report * sizeof(int));
    int* csrColIdxA_tmp = (int*)malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE* csrValA_tmp = (VALUE_TYPE*)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;
        int returnvalue;

        if (isReal)
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        } else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int*)malloc((m + 1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    csrColIdxA = (int*)malloc(nnzA * sizeof(int));
    csrValA = (VALUE_TYPE*)malloc(nnzA * sizeof(VALUE_TYPE));

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            } else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    } else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);

    // extract L with the unit-lower triangular sparsity structure of A
    int nnzL = 0;
    int* csrRowPtrL_tmp = (int*)malloc((m + 1) * sizeof(int));
    int* csrColIdxL_tmp = (int*)malloc(nnzA * sizeof(int));
    VALUE_TYPE* csrValL_tmp = (VALUE_TYPE*)malloc(nnzA * sizeof(VALUE_TYPE));

    int nnz_pointer = 0;
    csrRowPtrL_tmp[0] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i + 1]; j++)
        {
            if (csrColIdxA[j] < i)
            {
                csrColIdxL_tmp[nnz_pointer] = csrColIdxA[j];
                csrValL_tmp[nnz_pointer] = 1.0; //csrValA[j];
                nnz_pointer++;
            } else
            {
                break;
            }
        }

        csrColIdxL_tmp[nnz_pointer] = i;
        csrValL_tmp[nnz_pointer] = 1.0;
        nnz_pointer++;

        csrRowPtrL_tmp[i + 1] = nnz_pointer;
    }

    nnzL = csrRowPtrL_tmp[m];
    printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzL);

    csrColIdxL_tmp = (int*)realloc(csrColIdxL_tmp, sizeof(int) * nnzL);
    csrValL_tmp = (VALUE_TYPE*)realloc(csrValL_tmp, sizeof(VALUE_TYPE) * nnzL);

    printf("---------------------------------------------------------------------------------------------\n");
    
    int* RowPtrL_d, *ColIdxL_d;
    VALUE_TYPE* Val_d;

    cudaMalloc((void**)&RowPtrL_d, (n + 1) * sizeof(int));
    cudaMalloc((void**)&ColIdxL_d, nnzL * sizeof(int));
    cudaMalloc((void**)&Val_d, nnzL * sizeof(VALUE_TYPE));

    cudaMemcpy(RowPtrL_d, csrRowPtrL_tmp, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ColIdxL_d, csrColIdxL_tmp, nnzL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Val_d, csrValL_tmp, nnzL * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    int * iorder  = (int *) calloc(n,sizeof(int));

    
    double resultados[10];
    for (int i = 0; i < 10; i++) {
        std::chrono::high_resolution_clock::time_point start, end;
        start = std::chrono::high_resolution_clock::now();
        
        ordenar_filas(RowPtrL_d,ColIdxL_d,Val_d,n,iorder, 1);

        end = std::chrono::high_resolution_clock::now();    
        std::chrono::duration<double> diff = end - start;
        resultados[i] = diff.count();
    }
    
    double promedio = 0;
    for(int i = 0; i < 10; i++) {
        promedio += resultados[i];
    }
    promedio /= 10;

    double desviacion = 0;
    for(int i = 0; i < 10; i++) {
        desviacion += (resultados[i] - promedio) * (resultados[i] - promedio);
    }
    desviacion = sqrt(desviacion / 10);

    printf("PromedioGlobal: %f\n", promedio);
    printf("DesviaciónGlobal: %f\n", desviacion);

    int nwarps = ordenar_filas(RowPtrL_d,ColIdxL_d,Val_d,n,iorder, 10);
    printf("Number of warps: %i\n", nwarps);
    for(int i =0; i<n && i<20;i++)
        printf("Iorder[%i] = %i\n",i,iorder[i]);

    printf("Bye!\n");

    // done!
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);

    free(csrColIdxL_tmp);
    free(csrValL_tmp);
    free(csrRowPtrL_tmp);
    return 0;
}

