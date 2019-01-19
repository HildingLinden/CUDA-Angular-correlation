
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>

const int TILESIZE = 256;

// 2-point angular correlation
__global__
void DR_kernel(int n, float *d, double *r, unsigned int *DR) {
	// The thread id on the x-axis and y-axis
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * TILESIZE;

	if (x < n) {
		__shared__ unsigned int hist[720];
		//__shared__ double sharedR[TILESIZE * 2 + 1];

		// Right ascension of R
		//sharedR[threadIdx.x * 2] = r[(y + threadIdx.x) * 2];
		// Declination of R
		//sharedR[threadIdx.x * 2 + 1] = r[(y + threadIdx.x) * 2 + 1];

		// Right ascension and declination for the current element
		float asc1 = d[x * 2];
		float dec1 = d[x * 2 + 1];

		float decimalResult;
		// DR
		// n-y is the distance to the domain edge
		int nElements = min(n-y, TILESIZE);

		//__syncthreads();

		for (int j = 0; j < nElements; j++) {
			double asc2 = r[y + j * 2];
			double dec2 = r[y + j * 2 + 1];

			if (fabs(asc1-asc2) > 0.0001f && fabs(dec1-dec2) > 0.0001f) {
				decimalResult = acos(sin(dec1) * sin((float)dec2) + cos(dec1) * cos((float)dec2) * cos(asc1-(float)asc2));
				int resultIndex = floor(decimalResult/0.25);
				atomicAdd(&hist[resultIndex], 1);
			}
		}

		__syncthreads();

		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				atomicAdd(&DR[i], hist[i]);
			}
		}
	}
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
		printf("  Compute capability: %d.%d\n", props.major, props.minor);
		printf("  Shared memory per block: %zd\n", props.sharedMemPerBlock);
		printf("  Multiprocessor count: %d\n\n", props.multiProcessorCount);
	}

	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	// Reading both files and populating the arrays
	// Read file
	std::ifstream infile1("data_100k_arcmin.txt");

	// Get amount of coordinate pairs
	int nCoordinatePairs;
	infile1 >> nCoordinatePairs;
	printf("Found %d coordinate pairs in data\n", nCoordinatePairs);

	// Allocate memory for all pairs
	float *h_D = (float *)malloc(sizeof(float) * 2 * nCoordinatePairs);

	// Initialize the array of pairs
	float asc1, dec1;
	int index = 0;
	while (infile1 >> asc1 >> dec1) {
		if (index < nCoordinatePairs * 2) {
			h_D[index++] = asc1;
			h_D[index++] = dec1;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in data\n");
			exit(1);
		}
	}

	// Read file
	std::ifstream infile2("flat_100k_arcmin.txt");

	// Get amount of coordinate pairs
	int nCoordinatePairs2;
	infile2 >> nCoordinatePairs2;
	printf("Found %d coordinate pairs in flat\n", nCoordinatePairs2);

	// Allocate memory for all pairs
	double *h_R = (double *)malloc(sizeof(double) * 2 * nCoordinatePairs2);

	// Initialize the array of pairs
	double asc2, dec2;
	index = 0;
	while (infile2 >> asc2 >> dec2) {
		if (index < nCoordinatePairs2 * 2) {
			h_R[index++] = asc2;
			h_R[index++] = dec2;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in flat\n");
			exit(1);
		}
	}

	// Allocating and copying the input data to GPU
	float *d_D;
	double *d_R;

	cudaMalloc((void **)&d_D, nCoordinatePairs * 2 * sizeof(float));
	cudaMalloc((void **)&d_R, nCoordinatePairs * 2 * sizeof(double));

	cudaMemcpy(d_D, h_D, nCoordinatePairs * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R, nCoordinatePairs * 2 * sizeof(double), cudaMemcpyHostToDevice);

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

	int blockSize = TILESIZE;
	int gridSize = (nCoordinatePairs/4 + blockSize - 1) / blockSize;
	dim3 gridSize2D(gridSize, gridSize);
	DR_kernel<<<gridSize2D, blockSize>>>(nCoordinatePairs/4, d_D, d_R, d_DR);

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

	for (int i = 0; i < 20; i++) {
		printf("%d: %d\n", i, h_DR[i]);
	}

	// // Computing the difference
	// double *result;
	// result = (double *)malloc(sizeof(double) * 720);
	// printf("\nResult:\n");
	// for (int i = 0; i < 720; i++) {
	// 	if(h_RR[i] == 0) {
	// 		result[i] = 0.0;
	// 	} else {
	// 		result[i] = (h_DD[i] - 2 * h_DR[i] + h_RR[i]) / (double)h_RR[i];
	// 	}
	// 	printf("%d: %lf\n", i, result[i]);
	// }

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
