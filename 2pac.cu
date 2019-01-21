
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>

// 2-point angular correlation

const int ROWSPERTHREAD = 256;
const double RR_DEVICE_TO_HOST_RATIO = 0.90;

// Single-precision computation (is the precision enough?)
__global__ void DR_kernel(int nCols, int nRows, float *D, double *R, int *DR) {

	// The thread id on the x-axis and y-axis
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * ROWSPERTHREAD;

	// blockIdx.y * ROWSPERTHREAD

	if (x < nCols) {

		// Shared histogram for the thread block
		__shared__ int hist[720];

		// Thread number zero will initialize the shared memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				hist[i] = 0;
			}
		}

		__syncthreads();

		//__shared__ double sharedR[ROWSPERTHREAD * 2 + 1];

		// Right ascension of R
		//sharedR[threadIdx.x * 2] = r[(y + threadIdx.x) * 2];
		// Declination of R
		//sharedR[threadIdx.x * 2 + 1] = r[(y + threadIdx.x) * 2 + 1];

		// Right ascension and declination for the current element in D
		float asc1 = D[x * 2];
		float dec1 = D[x * 2 + 1];

		// Each thread calculates ROWSPERTHREAD elements of R or however many there are left to calculate
		int nElements = min(nRows-y, ROWSPERTHREAD);
		float decimalResult;

		for (int j = 0; j < nElements; j++) {
			double asc2 = R[y + j * 2];
			double dec2 = R[y + j * 2 + 1];

			if (fabs((double)asc1-asc2) > 0.0000001 || fabs((double)dec1-dec2) > 0.0000001) {
				decimalResult = acosf(sinf(dec1) * sinf(dec2) + cosf(dec1) * cosf(dec2) * cosf(asc1-(float)asc2));
				decimalResult *= 180/M_PI;
				int resultIndex = floor(decimalResult/0.25);
				atomicAdd(&hist[resultIndex], 1);
			}
		}

		__syncthreads();

		// Thread number zero will write the shared histogram to global device memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				atomicAdd(&DR[i], hist[i]);
			}
		}
	}
}

// All computation in single-precision
__global__ void DD_kernel(int n, float *D, int *DD) {

	// The thread id on the x-axis and y-axis
	//int x = blockIdx.y * ROWSPERTHREAD + blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * ROWSPERTHREAD;

	if (x < n && y + ROWSPERTHREAD > x) {

		// Shared histogram for the thread block
		__shared__ int hist[720];

		// Thread number zero will initialize the shared memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				hist[i] = 0;
			}
		}

		__syncthreads();

		// Right ascension and declination for the current element in D
		float asc1 = D[x * 2];
		float dec1 = D[x * 2 + 1];

		int offset = max(x-y+1, 0);

		// Each thread calculates ROWSPERTHREAD elements of D or however many there are left to calculate
		int nElements = min(n-y, ROWSPERTHREAD);

		float decimalResult;
		for (int j = offset; j < nElements; j++) {
			float asc2 = D[(y + j) * 2];
			float dec2 = D[(y + j) * 2 + 1];

			if (fabs(asc1-asc2) > 0.0000001 || fabs(dec1-dec2) > 0.0000001) {
				decimalResult = acosf(sinf(dec1) * sinf(dec2) + cosf(dec1) * cosf(dec2) * cosf(asc1-asc2));
				decimalResult *= 180/M_PI;
				int resultIndex = floor(decimalResult/0.25);
				atomicAdd(&hist[resultIndex], 1);
			}
		}

		__syncthreads();

		// Thread number zero will write the shared histogram to global device memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				atomicAdd(&DD[i], hist[i]);
			}
		}
	}
}

// All computation in double-precision
__global__ void RR_kernel(int xElements, int yElements, double *R, int *RR) {

	// The thread id on the x-axis and y-axis
	//int x = blockIdx.y * ROWSPERTHREAD + blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * ROWSPERTHREAD;

	if (x < xElements && y + ROWSPERTHREAD > x) {

		// Shared histogram for the thread block
		__shared__ int hist[720];

		// Thread number zero will initialize the shared memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				hist[i] = 0;
			}
		}

		__syncthreads();

		// Right ascension and declination for the current element in D
		double asc1 = R[x * 2];
		double dec1 = R[x * 2 + 1];

		int offset = max(x-y+1, 0);

		// Each thread calculates ROWSPERTHREAD elements of D or however many there are left to calculate
		int nElements = min(yElements-y, ROWSPERTHREAD);

		double decimalResult;
		for (int j = offset; j < nElements; j++) {
			double asc2 = R[(y + j) * 2];
			double dec2 = R[(y + j) * 2 + 1];

			if (fabs(asc1-asc2) > 0.0000001 || fabs(dec1-dec2) > 0.0000001) {
				decimalResult = acos(sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(asc1-asc2));
				decimalResult *= 180/M_PI;
				int resultIndex = floor(decimalResult/0.25);
				atomicAdd(&hist[resultIndex], 1);
			}
		}

		__syncthreads();

		// Thread number zero will write the shared histogram to global device memory
		if (threadIdx.x == 0) {
			for (int i = 0; i < 720; i++) {
				atomicAdd(&RR[i], hist[i]);
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
	double *h_R = (double *)malloc(nCoordinatePairsR * 2 * sizeof(double));

	if (h_D == NULL || h_R == NULL) printf("Allocating memory on host failed");


	int index = 0;

	// Initialize data
	float ascD, decD;
	while (infileD >> ascD >> decD) {
		if (index < nCoordinatePairsD * 2) {
			h_D[index++] = ascD;
			h_D[index++] = decD;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in data\n");
			exit(1);
		}
	}

	// Initialize synthetic
	double ascR, decR;
	index = 0;
	while (infileR >> ascR >> decR) {
		if (index < nCoordinatePairsR * 2) {
			h_R[index++] = ascR;
			h_R[index++] = decR;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in flat\n");
			exit(1);
		}
	}

	int *hostComputationResult = (int *)calloc(720, sizeof(int));

	float *d_D;
	double *d_R;
	int *h_DD, *h_DR, *h_RR;
	int *d_DD, *d_DR, *d_RR;

	#pragma omp parallel
	{
		// Device computation
		#pragma omp single nowait
		{
			printf("\nRunning host side on %d threads\n", omp_get_num_threads());

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

			// Allocating and copying the input data to device
			cudaMalloc((void **)&d_D, nCoordinatePairsD * 2 * sizeof(float));
			cudaMalloc((void **)&d_R, nCoordinatePairsR * 2 * sizeof(double));

			cudaMemcpy(d_D, h_D, nCoordinatePairsD * 2 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_R, h_R, nCoordinatePairsR * 2 * sizeof(double), cudaMemcpyHostToDevice);

			// Allocating and zero-initializing the result arrays on host
			h_DD = (int *)calloc(720, sizeof(int));
			h_DR = (int *)calloc(720, sizeof(int));
			h_RR = (int *)calloc(720, sizeof(int));

			// Allocating the result arrays on device
			cudaMalloc((void **)&d_DD, 720 * sizeof(int));
			cudaMalloc((void **)&d_DR, 720 * sizeof(int));
			cudaMalloc((void **)&d_RR, 720 * sizeof(int));

			// Copying the zero-initialized arrays to the GPU
			cudaMemcpy(d_DD, h_DD, 720 * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_DR, h_DR, 720 * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_RR, h_RR, 720 * sizeof(int), cudaMemcpyHostToDevice);

			// Calculating sizes and launching kernels
			int blockSize = 256;

			int gridSizeX = (nCoordinatePairsD + blockSize - 1) / blockSize;
			int gridSizeY = (nCoordinatePairsR + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
			dim3 gridSize2D(gridSizeX, gridSizeY);
			DR_kernel<<<gridSize2D, blockSize>>>(nCoordinatePairsD, nCoordinatePairsR, d_D, d_R, d_DR);

			gridSizeY = (nCoordinatePairsD + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
			dim3 DDGridSize2D(gridSizeX, gridSizeY);
			DD_kernel<<<DDGridSize2D, blockSize>>>(nCoordinatePairsD, d_D, d_DD);

			// Compute RR_DEVICE_TO_HOST_RATIO of columns in RR on device
			gridSizeX = (nCoordinatePairsR * RR_DEVICE_TO_HOST_RATIO + blockSize - 1) / blockSize;
			gridSizeY = (nCoordinatePairsR + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
			dim3 RRGridSize2D(gridSizeX, gridSizeY);
			RR_kernel<<<RRGridSize2D, blockSize>>>(nCoordinatePairsR * RR_DEVICE_TO_HOST_RATIO, nCoordinatePairsR, d_R, d_RR);
		}


		// Host computation

		// Each thread has a private histogram
		int *localHist = (int *)calloc(720, sizeof(int));
		
		#pragma omp for nowait
			// Compute 1 - RR_DEVICE_TO_HOST_RATIO of columns in RR on host
			for (int i = nCoordinatePairsR * RR_DEVICE_TO_HOST_RATIO; i < nCoordinatePairsR; i++) {

				double asc1 = h_R[i * 2];
				double dec1 = h_R[i * 2 + 1];

				for (int j = i+1; j < nCoordinatePairsR; j++) {

					double asc2 = h_R[j * 2];
					double dec2 = h_R[j * 2 + 1];

					if (fabs(asc1-asc2) > 0.0000001 || fabs(dec1-dec2) > 0.0000001) {
						double decimalResult = acos(sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(asc1-asc2));
						decimalResult *= 180/M_PI;
						int resultIndex = floor(decimalResult/0.25);
						if (resultIndex >= 0 || resultIndex <= 719) {
							localHist[resultIndex]++;
						} else {
							printf("Index %d and %d is incorrect with result %lf\n", i, j, decimalResult);
						}
					}
				}
			}

		// Each thread updates the global histogram
		#pragma omp critical
		{
			for (int i = 0; i < 720; i++) {
				hostComputationResult[i] += localHist[i];
			}
		}
	}

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
	cudaMemcpy(h_DD, d_DD, 720 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_DR, d_DR, 720 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_RR, d_RR, 720 * sizeof(int), cudaMemcpyDeviceToHost);

	// Combine device and host results for RR
	for (int i = 0; i < 720; i++) {
		h_RR[i] += hostComputationResult[i];
	}

	printf("\n");
	long long totalDD = 0;
	long long totalDR = 0;
	long long totalRR = 0;
	for (int i = 0; i < 720; i++) {
		totalDD += h_DD[i];
		totalDR += h_DR[i];
		totalRR += h_RR[i];
	}

	printf("Total computations\n");
	printf("DD: %lld\n", totalDD);
	printf("DR: %lld\n", totalDR);
	printf("RR: %lld\n", totalRR);

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
	//cudaFree(d_RR);
	free(h_D);
	free(h_R);
	free(h_DD);
	free(h_DR);
	free(h_RR);
}
