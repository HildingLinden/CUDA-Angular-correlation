
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>

// 2-point angular correlation

const int BLOCKSIZE = 256;
const int ROWSPERTHREAD = 256;

// Columns are D and rows are R
__global__ void DR_kernel(int nCols, int nRows, float *D, float *R, unsigned long long int *gHist) {

	// The thread id on the x-axis and y-axis
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * ROWSPERTHREAD;

	// If the thread is inside the domain
	if (x < nCols) {

		// Shared histogram for the thread block
		__shared__ unsigned int sHist[720];

		// Thread number zero will initialize the shared memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				sHist[i] = 0;
			}
		}

		__syncthreads();

		// Right ascension and declination in degrees for the current column
		float asc1 = D[x * 2];
		float dec1 = D[x * 2 + 1];

		// The amount of rows to be calculated is ROWSPERTHREAD or rows left in the domain, whichever is smaller
		int nElements = min(nRows-y, ROWSPERTHREAD);
		
		for (int j = 0; j < nElements; j++) {
			// Right ascension and declination degrees for the current row
			float asc2 = R[y + j * 2];
			float dec2 = R[y + j * 2 + 1];

			// Check if the coordinates are identical
			if (fabsf(asc1 - asc2) > 0.0000001f || fabsf(dec1 - dec2) > 0.0000001f) {
				// Compute the angle in radians
				float radianResult = acosf(sinf(dec1) * sinf(dec2) + cosf(dec1) * cosf(dec2) * cosf(asc1-asc2));
				// Convert to degrees
				float degreeResult = radianResult * 180/3.14159f;
				// Compute the bin
				int resultIndex = floor(degreeResult * 4.0f);
				// Increment the bin in the shared histogram
				atomicAdd(&sHist[resultIndex], 1);
			} else {
				//printf("Same coordinates in DR\n");
				atomicAdd(&sHist[0], 1);
			}
		}

		__syncthreads();

		// Thread number zero will write the shared histogram to global device memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				// Update the global histogram with the shared histogram
				atomicAdd(&gHist[i], sHist[i]);
			}
		}
	}
}

// All computation in single-precision
__global__ void DD_or_RR_kernel(int nCols, int nRows, float *arr, unsigned long long int *gHist) {

	// The thread id on the x-axis and y-axis
	//int x = blockIdx.y * ROWSPERTHREAD + blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * ROWSPERTHREAD;

	// If the column is inside the domain and the last row of the thread should be computed
	if (x < nCols && y + ROWSPERTHREAD > x) {

		// Shared histogram for the thread block
		__shared__ unsigned int sHist[720];

		// Thread number zero will initialize the shared memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				sHist[i] = 0;
			}
		}

		__syncthreads();

		// Right ascension and declination in degrees for the current column
		float asc1 = arr[x * 2];
		float dec1 = arr[x * 2 + 1];

		// Offset is at which row to start computing
		int offset = max(x-y+1, 0);

		// The amount of rows to be calculated is ROWSPERTHREAD or rows left in the domain, whichever is smaller
		int nElements = min(nRows-y, ROWSPERTHREAD);
		
		for (int j = offset; j < nElements; j++) {
			// Right ascension and declination in degrees for the current row
			float asc2 = arr[y + j * 2];
			float dec2 = arr[y + j * 2 + 1];

			// Check if the coordinates are identical
			if (fabsf(asc1 - asc2) > 0.0000001f || fabsf(dec1 - dec2) > 0.0000001f) {
				// Compute the angle in radians
				float radianResult = acosf(sinf(dec1) * sinf(dec2) + cosf(dec1) * cosf(dec2) * cosf(asc1-asc2));
				// Convert to degrees
				float degreeResult = radianResult * 180/3.14159f;
				// Compute the bin
				int resultIndex = floor(degreeResult * 4.0f);
				// Increment the bin in the shared histogram
				atomicAdd(&sHist[resultIndex], 2);
			} else {
				//printf("Same coordinates in DD or RR\n");
				// Add two we compute the angle between a pair only once
				atomicAdd(&sHist[0], 2);
			}
		}

		__syncthreads();

		// Thread number zero will write the shared histogram to global device memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				// Update the global histogram with the shared histogram
				atomicAdd(&gHist[i], sHist[i]);
			}
		}
	}
}

int main(void) {
	
	// Read real data file
	std::ifstream infileD("data_100k_arcmin.txt");

	// Get amount of coordinate pairs
	int nCoordinatePairsD;
	infileD >> nCoordinatePairsD;
	printf("Found %d coordinate pairs in data\n", nCoordinatePairsD);

	// Allocate memory for real data on host
	float *h_D = (float *)malloc(nCoordinatePairsD * 2 * sizeof(float));

	// Read synthetic data file
	std::ifstream infileR("flat_100k_arcmin.txt");

	// Get amount of coordinate pairs
	int nCoordinatePairsR;
	infileR >> nCoordinatePairsR;
	printf("Found %d coordinate pairs in flat\n", nCoordinatePairsR);

	// Allocate memory for synthetic data on host
	float *h_R = (float *)malloc(nCoordinatePairsR * 2 * sizeof(float));

	if (h_D == NULL || h_R == NULL) printf("Allocating memory on host failed");

	int index = 0;

	// Read, convert from arc minutes to degrees and store in D
	float ascD, decD;
	while (infileD >> ascD >> decD) {
		if (index < nCoordinatePairsD * 2) {
			h_D[index++] = ascD * 1/60 * M_PI/180;
			h_D[index++] = decD * 1/60 * M_PI/180;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in data\n");
			exit(1);
		}
	}

	// Read, convert from arc minutes to degrees and store in R
	float ascR, decR;
	index = 0;
	while (infileR >> ascR >> decR) {
		if (index < nCoordinatePairsR * 2) {
			h_R[index++] = ascR * 1/60 * M_PI/180;
			h_R[index++] = decR * 1/60 * M_PI/180;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in flat\n");
			exit(1);
		}
	}

	float *d_D;
	float *d_R;
	unsigned long long int *h_DD, *h_DR, *h_RR;
	unsigned long long int *d_DD, *d_DR, *d_RR;

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

	// Allocating and copying the input data to device
	cudaMalloc((void **)&d_D, nCoordinatePairsD * 2 * sizeof(float) * 100);
	cudaMalloc((void **)&d_R, nCoordinatePairsR * 2 * sizeof(float) * 100);

	cudaMemcpy(d_D, h_D, nCoordinatePairsD * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R, nCoordinatePairsR * 2 * sizeof(float), cudaMemcpyHostToDevice);

	// Allocating and zero-initializing the result arrays on host
	h_DD = (unsigned long long int *)calloc(720, sizeof(unsigned long long int));
	h_DR = (unsigned long long int *)calloc(720, sizeof(unsigned long long int));
	h_RR = (unsigned long long int *)calloc(720, sizeof(unsigned long long int));

	// Allocating the result arrays on device
	cudaMalloc((void **)&d_DD, 720 * sizeof(unsigned long long int));
	cudaMalloc((void **)&d_DR, 720 * sizeof(unsigned long long int));
	cudaMalloc((void **)&d_RR, 720 * sizeof(unsigned long long int));

	// Copying the zero-initialized arrays to the GPU
	cudaMemcpy(d_DD, h_DD, 720 * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_DR, h_DR, 720 * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_RR, h_RR, 720 * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

	// Device kernel for DR
	int gridSizeX = (nCoordinatePairsD + BLOCKSIZE - 1) / BLOCKSIZE;
	int gridSizeY = (nCoordinatePairsR + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
	dim3 gridSize2D(gridSizeX, gridSizeY);
	DR_kernel<<<gridSize2D, BLOCKSIZE>>>(nCoordinatePairsD, nCoordinatePairsR, d_D, d_R, d_DR);

	// Device kernel for DD
	gridSizeY = (nCoordinatePairsD + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
	dim3 DDGridSize2D(gridSizeX, gridSizeY);
	DD_or_RR_kernel<<<DDGridSize2D, BLOCKSIZE>>>(nCoordinatePairsD, nCoordinatePairsD, d_D, d_DD);

	// Device kernel for RR
	gridSizeX = (nCoordinatePairsR + BLOCKSIZE - 1) / BLOCKSIZE;
	gridSizeY = (nCoordinatePairsR + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
	dim3 RRGridSize2D(gridSizeX, gridSizeY);
	DD_or_RR_kernel<<<RRGridSize2D, BLOCKSIZE>>>(nCoordinatePairsR, nCoordinatePairsR, d_R, d_RR);

	// Checking for errors
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();

	if (errSync != cudaSuccess) {
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	}
	if (errAsync != cudaSuccess) {
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	}

	// Copying the result from device to host
	// cudaMemcpy has an implicit barrier
	cudaMemcpy(h_DD, d_DD, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_DR, d_DR, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_RR, d_RR, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	// Add the number of galaxies in D to the first bin in DD and respectively in R and RR since we start
	// computing for the index + 1
	h_DD[0] += nCoordinatePairsD;
	h_RR[0] += nCoordinatePairsR;

	printf("\n");
	long long totalDD = 0;
	long long totalDR = 0;
	long long totalRR = 0;
	for (int i = 0; i < 720; i++) {
		totalDD += h_DD[i];
		totalDR += h_DR[i];
		totalRR += h_RR[i];
	}

	printf("Total count in histograms\n");
	printf("DD: %lld\n", totalDD);
	printf("DR: %lld\n", totalDR);
	printf("RR: %lld\n", totalRR);

	// Computing the difference
	//double result[720];
	printf("\nResult:\n");
	for (int i = 0; i < 10; i++) {
		//printf("%d: DD: %llu, DR: %llu, RR: %llu\n", i, h_DD[i], h_DR[i], h_RR[i]);
		printf("  %lf\n", ((double)h_DD[i] - 2.0 * (double)h_DR[i] + (double)h_RR[i]) / (double)h_RR[i]);
		// if(h_RR[i] == 0) {
		// 	result[i] = 0.0;
		// } else {
		// 	result[i] = (h_DD[i] - 2 * h_DR[i] + h_RR[i]) / (double)h_RR[i];
		// }
		// printf("%d: %lf\n", i, result[i]);
	}

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
