#!/bin/bash
#SBATCH --job-name=gpgpu_practico
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00

#SBATCH --partition=besteffort
# SBATCH --partition=normal

#SBATCH --qos=besteffort_gpu
# SBATCH --qos=gpu

#SBATCH --gres=gpu:p100:1
# #SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriel.kryger@fing.edu.uy
#SBATCH -o consola.out

export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64

# Compilar el programa CUDA
nvcc ej1a.cu -o ej1a.out
nvcc ej1b.cu -o ej1b.out
nvcc ej2a.cu -o ej2a.out
nvcc ej2b.cu -o ej2b.out
nvcc ej3a.cu -o ej3a.out
nvcc ej3b.cu -o ej3b.out

# Ejecutar el programa CUDA con el archivo secreto.txt como argumento
echo "------------------------------------\n"
echo "ej1a"
time ./ej1a.out
echo "ej1b"
time ./ej1b.out
echo "ej2a"
time ./ej2a.out
echo "ej2b"
time ./ej2b.out
echo "ej3a"
time ./ej3a.out
echo "ej3b"
time ./ej3b.out
echo "------------------------------------\n"


