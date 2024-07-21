all:
	nvcc -arch=sm_60 main_1.cu blur_1_quicksort.cu  -o blur_1 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w 
	nvcc -arch=sm_60 main_2.cu  blur_2_radixsort.cu -o blur_2 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	nvcc -arch=sm_60 main_3.cu  blur_3_thrust.cu -o blur_3 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	nvcc -arch=sm_60 main_3.cu  blur_3m_thrust.cu -o blur_3m -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	nvcc -arch=sm_60 main_4.cu  blur_4_paralelismo.cu -o blur_4 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	nvcc -arch=sm_60 main_5.cu  blur_5_unificado.cu -o blur_5 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	./blur_1 1.pgm
	./blur_2 1.pgm
	./blur_3 1.pgm
	./blur_3m 1.pgm
	./blur_4 1.pgm
	./blur_5 1.pgm
	./blur_1 fing1_ruido.pgm 
	./blur_2 fing1_ruido.pgm
	./blur_3 fing1_ruido.pgm
	./blur_3m fing1_ruido.pgm
	./blur_4 fing1_ruido.pgm
	./blur_5 fing1_ruido.pgm
	./blur_1 Facultad_de_Ingeniería_UDELAR.pgm
	./blur_2 Facultad_de_Ingeniería_UDELAR.pgm
	./blur_3 Facultad_de_Ingeniería_UDELAR.pgm
	./blur_3m Facultad_de_Ingeniería_UDELAR.pgm
	./blur_4 Facultad_de_Ingeniería_UDELAR.pgm
	./blur_5 Facultad_de_Ingeniería_UDELAR.pgm


clean:
	rm -f blur_1 blur_2 blur_3 blur_3m blur_4 blur_5

1:
	nvcc -arch=sm_60 main_1.cu blur_1_quicksort.cu  -o blur_1 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w 
	./blur_1 1.pgm
	./blur_1 fing1_ruido.pgm
	./blur_1 Facultad_de_Ingeniería_UDELAR.pgm

2:
	nvcc -arch=sm_60 main_2.cu  blur_2_radixsort.cu -o blur_2 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	./blur_2 1.pgm
	./blur_2 fing1_ruido.pgm
	./blur_2 Facultad_de_Ingeniería_UDELAR.pgm

3:
	nvcc -arch=sm_60 main_3.cu  blur_3_thrust.cu -o blur_3 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	./blur_3 1.pgm
	./blur_3 fing1_ruido.pgm
	./blur_3 Facultad_de_Ingeniería_UDELAR.pgm

4:
	nvcc -arch=sm_60 main_4.cu  blur_4_paralelismo.cu -o blur_4 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	./blur_4 1.pgm
	./blur_4 fing1_ruido.pgm
	./blur_4 Facultad_de_Ingeniería_UDELAR.pgm

5: 	
	nvcc -arch=sm_60 main_5.cu  blur_5_unificado.cu -o blur_5 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	./blur_5 1.pgm
	./blur_5 fing1_ruido.pgm
	./blur_5 Facultad_de_Ingeniería_UDELAR.pgm

3m:
	nvcc -arch=sm_60 main_3m.cu blur_3m_thrust.cu -o blur_3m -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt -w -std=c++11
	./blur_3m 1.pgm
	./blur_3m fing1_ruido.pgm
	./blur_3m Facultad_de_Ingeniería_UDELAR.pgm

