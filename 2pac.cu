#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>

// 2-point angular correlation

__global__
void kernel(int n, double *d, double *r, unsigned int *DD, unsigned int *DR, unsigned int *RR) {
	int threadIndex = blockIdx.x*blockDim.x + threadIdx.x;

	if (threadIndex < n) {
		// Right ascension and declination for the current element
		double asc = d[threadIndex*2];
		double dec = d[threadIndex*2+1];

		// Alpha and delta values for the current element
		double alpha1 = asc;
		double delta1 = dec;

		double floatResult;
		// DD
		for (int j = threadIndex+1; j < n; j++) {
			double asc = d[j*2];
			double dec = d[j*2+1];

			double alpha2 = asc;
			double delta2 = dec;

			if (alpha1 == alpha2 && delta1 == delta2) {
				printf("Index %d and index %d of D contains the same coordinates\n", threadIndex, j);
			} else {
				floatResult = acos(sin(delta1) * sin(delta2) + cos(delta1) * cos(delta2) * cos(alpha1-alpha2));
				int resultIndex = floor(floatResult/0.25);
				if (resultIndex >= 0) {
					atomicAdd(&DD[resultIndex], 1);
				} else {
					printf("Result of DD incorrect");
				}
			}
		}

		// DR
		for (int j = 0; j < n; j++) {
			double asc = r[j*2];
			double dec = r[j*2+1];

			double alpha2 = asc;
			double delta2 = dec;

			if (alpha1 == alpha2 && delta1 == delta2) {
				printf("Index %d of D and index %d of R contains the same coordinates\n", threadIndex, j);
			} else {
				floatResult = acos(sin(delta1) * sin(delta2) + cos(delta1) * cos(delta2) * cos(alpha1-alpha2));
				int resultIndex = floor(floatResult/0.25);
				if (resultIndex >= 0) {
					atomicAdd(&DD[resultIndex], 1);
				} else {
					printf("Result of DR incorrect");
				}
			}
		}

		// RR
		asc = r[threadIndex*2];
		dec = r[threadIndex*2+1];

		alpha1 = asc;
		delta1 = dec;

		for (int j = threadIndex+1; j < n; j++) {
			double asc = r[j*2];
			double dec = r[j*2+1];

			double alpha2 = asc;
			double delta2 = dec;

			if (alpha1 == alpha2 && delta1 == delta2) {
				printf("Index %d and index %d of R contains the same coordinates\n", threadIndex, j);
			} else {
				floatResult = acos(sin(delta1) * sin(delta2) + cos(delta1) * cos(delta2) * cos(alpha1-alpha2));
				int resultIndex = floor(floatResult/0.25);
				if (resultIndex >= 0) {
					atomicAdd(&DD[resultIndex], 1);
				} else {
					printf("Result of RR incorrect");
				}
			}
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

	/*printf("\n");
	for (int i = 0; i < 720; i++) {
		printf("DD[%d]: %d ", i, h_DD[i]);
		printf("DR[%d]: %d ", i, h_DR[i]);
		printf("RR[%d]: %d\n", i, h_RR[i]);
	}*/ 

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
