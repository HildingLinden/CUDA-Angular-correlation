2pac: 2pac.cu
	nvcc -O3 -ccbin=gcc-7 -Xcompiler -ffast_math,-fopenmp -arch=sm_61 -Xptxas=-v -use_fast_math 2pac.cu -o 2pac