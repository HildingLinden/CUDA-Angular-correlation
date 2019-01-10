#include <stdio.h>
#include <fstream>
#include <iostream>

// 2-point angular correlation

__global__
void saxpy(int n, float a, float *x, float *y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

struct CoordinatePair {
	float rightAscension;
	float declination;
};

CoordinatePair * readFile(std::string name) {
	// Read file
	std::ifstream infile(name.c_str());

	// Get amount of coordinate pairs
	int nCoordinatePairs;
	infile >> nCoordinatePairs;
	printf("Found %d coordinate pairs in %s\n", nCoordinatePairs, name.c_str());

	// Allocate memory for all pairs
	CoordinatePair *arr = (CoordinatePair *)malloc(sizeof(CoordinatePair) * nCoordinatePairs);

	// Initialize the array of pairs
	float asc, dec;
	int index = 0;
	while (infile >> asc >> dec) {
		if (index < nCoordinatePairs) {
			arr[index].rightAscension = asc;
			arr[index++].declination = dec;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in %s\n", name.c_str());
			exit(1);
		}
	}

	return arr;
}

int main(void) {
	CoordinatePair *D = readFile("data_100k_arcmin.txt");
	CoordinatePair *R = readFile("flat_100k_arcmin.txt");

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		printf("\nDevice nr.: %d\n", i);
		printf("  Device name: %s\n", props.name);
		printf("  Memory clock rate: (MHz) %lf\n", props.memoryClockRate/1000.0);
		printf("  Memory bus width (bits): %d\n", props.memoryBusWidth);
		printf("  Peak memory bandwith (GB/s): %f\n", 2.0*props.memoryClockRate*(props.memoryBusWidth/8)/1.0e6);
		printf("  Compute capability: %d.%d\n\n", props.major, props.minor);
	}

	int n = 1 << 20;
	float *x, *y, *d_x, *d_y;
	// Using pinned host memory to speed up transfer
	cudaMallocHost((void **)&x, n*sizeof(float));
	cudaMallocHost((void **)&y, n*sizeof(float));

	cudaMalloc((void **)&d_x, n*sizeof(float));
	cudaMalloc((void **)&d_y, n*sizeof(float));

	for (int i = 0; i < n; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

	saxpy<<<(n+255)/256, 256>>>(n, 2.0f, d_x, d_y);

	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();

	if (errSync != cudaSuccess) {
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	}
	if (errAsync != cudaSuccess) {
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	}

	cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < n; ++i) {
		maxError = max(maxError, abs(y[i]-4.0f));
	}
		printf("Max error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFreeHost(x);
	cudaFreeHost(y);
}