# make -B to force

2pac: 2pac.cu
	nvcc -Ofast -ccbin=g++-7 -Xcompiler -ffast-math,-fopenmp -arch=sm_61 -Xptxas=-v -use_fast_math 2pac.cu -o 2pac

2pac_debug: 2pac.cu
	nvcc -g -ccbin=g++-7 -Xcompiler -ffast-math,-fopenmp -arch=sm_61 -Xptxas=-v -use_fast_math 2pac.cu -o 2pac

avx: avx_2pac_test.cpp
	g++-7 -Ofast -fopt-info-vec -Wall -Wextra -mavx2 -lm -march=native avx_2pac_test.cpp -o avx

avx_debug: avx_2pac_test.cpp
	g++-7 -g -fopt-info-vec -Wall -Wextra -mavx2 -lm -march=native avx_2pac_test.cpp -o avx