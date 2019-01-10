#include <stdio.h>
#include <fstream>
#include <iostream>

// 2-point angular correlation

__global__
void kernel(int n, float *x, float *y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		y[i] = x[i] + y[i];
	}
}

float * readFile(std::string name) {
	// Read file
	std::ifstream infile(name.c_str());

	// Get amount of coordinate pairs
	int nCoordinatePairs;
	infile >> nCoordinatePairs;
	printf("Found %d coordinate pairs in %s\n", nCoordinatePairs, name.c_str());

	// Allocate memory for all pairs
	float *arr = (float *)malloc(sizeof(float) * 2 * nCoordinatePairs);

	// Initialize the array of pairs
	float asc, dec;
	int index = 0;
	while (infile >> asc >> dec) {
		if (index < nCoordinatePairs * 2) {
			arr[index++] = asc;
			arr[index++] = dec;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in %s\n", name.c_str());
			exit(1);
		}
	}

	return arr;
}

int main(void) {
	// Info about the GPU
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

	// Reading both files and populating the arrays
	float *D = readFile("data_100k_arcmin.txt");
	float *R = readFile("flat_100k_arcmin.txt");

	// Get amount of pairs to be able to allocate memory on device
	std::ifstream infile("data_100k_arcmin.txt");
	int nCoordinatePairs;
	infile >> nCoordinatePairs;
	int size = nCoordinatePairs * 2 * sizeof(float);

	float *d_D, *d_R;

	cudaMalloc((void **)&d_D, size);
	cudaMalloc((void **)&d_R, size);

	cudaMemcpy(d_D, D, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);

	kernel<<<(nCoordinatePairs+255)/256, 256>>>(nCoordinatePairs, d_D, d_R);

	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();

	if (errSync != cudaSuccess) {
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	}
	if (errAsync != cudaSuccess) {
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	}

	cudaMemcpy(R, d_R, size, cudaMemcpyDeviceToHost);

	printf("\n%f\n", R[0]);
	// float maxError = 0.0f;
	// for (int i = 0; i < n; ++i) {
	// 	maxError = max(maxError, abs(y[i]-4.0f));
	// }
	
	// printf("Max error: %f\n", maxError);

	cudaFree(d_D);
	cudaFree(d_R);
	free(D);
	free(R);
}