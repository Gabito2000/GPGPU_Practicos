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

# Ejecutar el programa CUDA con el archivo secreto.txt como argumento
./ej1a.out
