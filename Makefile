2pac: 2pac.cu
	nvcc -O3 -ccbin=g++-7 -Xcompiler -ffast-math,-fopenmp -arch=sm_61 -Xptxas=-v -use_fast_math 2pac.cu -o 2pac

avx: avx_2pac_test.cpp
	g++-7 -Ofast -fopt-info-vec -Wall -Wextra -mavx2 -lm -march=native avx_2pac_test.cpp -o avx