all:
	nvcc -arch=sm_60 main.cu blur_1_quicksort.cu  -o blur_1 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt
	nvcc -arch=sm_60 main.cu  blur_2_radixsort.cu -o blur_2 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt
	nvcc -arch=sm_60 main.cu  blur_3_cub.cu -o blur_3 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt
	
