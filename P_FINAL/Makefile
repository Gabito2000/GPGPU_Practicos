
all:
	nvcc -arch=sm_60 main_1.cu blur_1_quicksort.cu  -o blur_1 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w 
	nvcc -arch=sm_60 main_2.cu  blur_2_radixsort.cu -o blur_2 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	nvcc -arch=sm_60 main_3.cu  blur_3_thrust.cu -o blur_3 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	./blur_1 fing1_ruido.pgm 
	./blur_2 fing1_ruido.pgm
	./blur_3 fing1_ruido.pgm

