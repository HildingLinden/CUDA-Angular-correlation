
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>

const int TILESIZE = 512;

// 2-point angular correlation
__global__
void DR_kernel(int n, double *d, double *r, unsigned int *DR) {
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
		double asc1 = d[x * 2];
		double dec1 = d[x * 2 + 1];

		double doubleResult;
		// DR
		// n-y is the distance to the domain edge
		int nElements = min(n-y, TILESIZE);

		//__syncthreads();

		for (int j = 0; j < nElements; j++) {
			double asc2 = r[y + j * 2];
			double dec2 = r[y + j * 2 + 1];

			if (fabs(asc1-asc2) > 0.0001f && fabs(dec1-dec2) > 0.0001f) {
				doubleResult = acos(sin((float)dec1) * sin((float)dec2) + cos((float)dec1) * cos((float)dec2) * cos((float)asc1-(float)asc2));
				int resultIndex = floor(doubleResult/0.25);
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
		printf("  Compute capability: %d.%d\n", props.major, props.minor);
		printf("  Shared memory per block: %zd\n", props.sharedMemPerBlock);
		printf("  Multiprocessor count: %d\n\n", props.multiProcessorCount);
	}

	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

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

	int blockSize = TILESIZE;
	int gridSize = (nCoordinatePairs + blockSize - 1) / blockSize;
	dim3 gridSize2D(gridSize, gridSize);
	DR_kernel<<<gridSize2D, blockSize>>>(nCoordinatePairs, d_D, d_R, d_DR);

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
