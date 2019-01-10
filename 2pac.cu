#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>

// 2-point angular correlation

__global__
void kernel(int n, double *x, double *y, unsigned int *DD, unsigned int *DR, unsigned int *RR) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pairIndex = i * 2;

	if (pairIndex < n) {
		double asc = x[pairIndex];
		double dec = x[pairIndex+1];

		double alpha1 = asc;
		double beta1 = 90 - dec;

		double floatResult;

		for (int j = 0; j < n; j++) {
			double alpha2 = x[j];
			double beta2 = x[j+1];

			floatResult = acos(sin(beta1)*sin(beta2)+cos(beta1)*cos(beta2)*cos(alpha1-alpha2));
			int resultIndex = floor(floatResult/0.25);
			atomicAdd(&DD[resultIndex], 1);
		}
	}
}

double * readFile(std::string name) {
	// Read file
	std::ifstream infile(name.c_str());

	// Get amount of coordinate pairs
	int nCoordinatePairs;
	infile >> nCoordinatePairs;
	printf("Found %d coordinate pairs in %s\n", nCoordinatePairs, name.c_str());

	// Allocate memory for all pairs
	double *arr = (double *)malloc(sizeof(double) * 2 * nCoordinatePairs);

	// Initialize the array of pairs
	double asc, dec;
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
	double *h_D = readFile("data_100k_arcmin.txt");
	double *h_R = readFile("flat_100k_arcmin.txt");

	// Get amount of pairs to be able to allocate memory on device
	std::ifstream infile("data_100k_arcmin.txt");
	int nCoordinatePairs;
	infile >> nCoordinatePairs;
	int inputSize = nCoordinatePairs * 2 * sizeof(double);

	// Allocating and copying the input data to GPU
	double *d_D, *d_R;

	cudaMalloc((void **)&d_D, inputSize);
	cudaMalloc((void **)&d_R, inputSize);

	cudaMemcpy(d_D, h_D, inputSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R, inputSize, cudaMemcpyHostToDevice);

	int resultSize = 720 * sizeof(unsigned int);

	// Allocating and zero-initializing the result arrays on CPU
	unsigned int *h_DD, *h_DR, *h_RR;
	h_DD = (unsigned int *)calloc(resultSize, sizeof(unsigned int));
	h_DR = (unsigned int *)calloc(resultSize, sizeof(unsigned int));
	h_RR = (unsigned int *)calloc(resultSize, sizeof(unsigned int));

	// Allocating the result arrays on GPU
	unsigned int *d_DD, *d_DR, *d_RR;
	cudaMalloc((void **)&d_DD, resultSize);	
	cudaMalloc((void **)&d_DR, resultSize);
	cudaMalloc((void **)&d_RR, resultSize);

	// Copying the zero-initialized arrays to the GPU
	cudaMemcpy(d_DD, h_DD, resultSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_DR, h_DR, resultSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_RR, h_RR, resultSize, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (nCoordinatePairs + blockSize - 1) / blockSize;
	kernel<<<gridSize, blockSize>>>(nCoordinatePairs, d_D, d_R, d_DD, d_DR, d_RR);

	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();

	if (errSync != cudaSuccess) {
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	}
	if (errAsync != cudaSuccess) {
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	}

	// cudaMemcpy has an implicit barrier
	// Copying back the result
	cudaMemcpy(h_DD, d_DD, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_DR, d_DR, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_RR, d_RR, resultSize, cudaMemcpyDeviceToHost);

	printf("\n%d %d %d\n", h_DD[0], h_DR[0], h_RR[0]);
	// double maxError = 0.0f;
	// for (int i = 0; i < n; ++i) {
	// 	maxError = max(maxError, abs(y[i]-4.0f));
	// }
	
	// printf("Max error: %f\n", maxError);

	cudaFree(d_D);
	cudaFree(d_R);
	cudaFree(d_DD);
	cudaFree(d_DR);
	cudaFree(d_RR);
	free(h_D);
	free(h_R);
	free(h_DD);
	free(h_DR);
	free(h_RR);
}