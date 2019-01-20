
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>

// 2-point angular correlation

const int ROWSPERTHREAD = 256;

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
				int resultIndex = floor(decimalResult/0.25);
				atomicAdd(&hist[resultIndex], 1);
			} else {
				printf("Coordinate %d of D and %d of R are identical\n", x, y+j);
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
				int resultIndex = floor(decimalResult/0.25);
				atomicAdd(&hist[resultIndex], 1);
			} else {
				printf("%d: %f - %f and %d: %f - %f \n", x, D[x*2], D[x*2+1], y+j, D[(y+j)*2], D[(y+j)*2+1]);
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
__global__ void RR_kernel(int n, double *R, int *RR) {

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
		double asc1 = R[x * 2];
		double dec1 = R[x * 2 + 1];

		int offset = max(x-y+1, 0);

		// Each thread calculates ROWSPERTHREAD elements of D or however many there are left to calculate
		int nElements = min(n-y, ROWSPERTHREAD);

		double decimalResult;
		for (int j = offset; j < nElements; j++) {
			double asc2 = R[(y + j) * 2];
			double dec2 = R[(y + j) * 2 + 1];

			if (fabs(asc1-asc2) > 0.0000001 || fabs(dec1-dec2) > 0.0000001) {
				decimalResult = acos(sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(asc1-asc2));
				int resultIndex = floor(decimalResult/0.25);
				atomicAdd(&hist[resultIndex], 1);
			} else {
				printf("%d: %lf - %lf and %d: %lf - %lf \n", x, R[x*2], R[x*2+1], y+j, R[(y+j)*2], R[(y+j)*2+1]);
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


	// Allocating and copying the input data to device
	float *d_D;
	double *d_R;

	cudaMalloc((void **)&d_D, nCoordinatePairsD * 2 * sizeof(float));
	cudaMalloc((void **)&d_R, nCoordinatePairsR * 2 * sizeof(double));

	cudaMemcpy(d_D, h_D, nCoordinatePairsD * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R, nCoordinatePairsR * 2 * sizeof(double), cudaMemcpyHostToDevice);


	// Allocating and zero-initializing the result arrays on host
	int *h_DD, *h_DR, *h_RR;
	h_DD = (int *)calloc(720, sizeof(int));
	h_DR = (int *)calloc(720, sizeof(int));
	h_RR = (int *)calloc(720, sizeof(int));

	// Allocating the result arrays on device
	int *d_DD, *d_DR, *d_RR;
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

	gridSizeX = (nCoordinatePairsR + blockSize - 1) / blockSize;
	gridSizeY = (nCoordinatePairsR + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
	dim3 RRGridSize2D(gridSizeX, gridSizeY);
	RR_kernel<<<RRGridSize2D, blockSize>>>(nCoordinatePairsR, d_R, d_RR);

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

	printf("\n");
	long long total = 0;
	for (int i = 0; i < 20; i++) {
		printf("%d: %u\n", i, h_DR[i]);
		total += h_DR[i];
	}

	printf("\nTotal: %lld\n", total);

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
