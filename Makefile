CXX=g++-7
NVCC=nvcc
CXXFLAGS= "-fopt-info-vec -fopenmp -Ofast -Wall -Wextra -mfma -march=native"
CUDAFLAGS= -arch=sm_61 -Xptxas=-v -use_fast_math
LIBS= -lm

all: 2pac.cu
	$(NVCC) $(CUDAFLAGS) -ccbin=$(CXX) -Xcompiler $(CXXFLAGS) $(LIBS) 2pac.cu -o 2pac