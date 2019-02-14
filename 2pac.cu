#include <stdlib.h>			/* atof, malloc, alloc, exit */
#include <stdio.h>			/* printf */
#include <fstream>			/* ifstream */
#include <iostream>			/* err */
#include <math.h>			/* M_PI */
#include <omp.h>			/* OpenMP */
#include <immintrin.h> 		/* AVX */
#include <cmath>			/* lround */
#include "avx_mathfun.h"	/* sincos256_ps */

// 2-point angular correlation

/* https://stackoverflow.com/a/46991254/10105352 */
inline __m256 acosv(__m256 x) {
    __m256 xp = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    // main shape
    __m256 one = _mm256_set1_ps(1.0);
    __m256 t = _mm256_sqrt_ps(_mm256_sub_ps(one, xp));
    // polynomial correction factor based on xp
    __m256 c3 = _mm256_set1_ps(-0.02007522);
    __m256 c2 = _mm256_fmadd_ps(xp, c3, _mm256_set1_ps(0.07590315));
    __m256 c1 = _mm256_fmadd_ps(xp, c2, _mm256_set1_ps(-0.2126757));
    __m256 c0 = _mm256_fmadd_ps(xp, c1, _mm256_set1_ps(1.5707963267948966));
    // positive result
    __m256 p = _mm256_mul_ps(t, c0);
    // correct for negative x
    __m256 n = _mm256_sub_ps(_mm256_set1_ps(3.14159265359), p);
    return _mm256_blendv_ps(p, n, x);
}

const int BLOCKSIZE = 256;
const int ROWSPERTHREAD = 256;
double DD_DEVICE_TO_HOST_RATIO = 0.95;

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

			// Compute the intermediate value
			float tmp = sinf(dec1) * sinf(dec2) + cosf(dec1) * cosf(dec2) * cosf(asc1-asc2);

			// Clamp it to -1, 1
			tmp = fminf(tmp, 1.0f);
			tmp = fmaxf(tmp, -1.0f);

			// Compute the angle in radians
			float radianResult = acosf(tmp);

			// Convert to degrees
			float degreeResult = radianResult * 180.0f/3.14159f;

			// Compute the bin index
			int resultIndex = floor(degreeResult * 4.0f);

			// Increment the bin in the shared histogram
			atomicAdd(&sHist[resultIndex], 1);

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
			float asc2 = arr[(y + j) * 2];
			float dec2 = arr[(y + j) * 2 + 1];

			// Compute the intermediate value
			float tmp = sinf(dec1) * sinf(dec2) + cosf(dec1) * cosf(dec2) * cosf(asc1-asc2);

			// Clamp it to -1, 1
			tmp = fminf(tmp, 1.0f);
			tmp = fmaxf(tmp, -1.0f);

			// Compute the angle in radians
			float radianResult = acosf(tmp);

			// Convert to degrees
			float degreeResult = radianResult * 180.0f/3.14159f;

			// Compute the bin index
			int resultIndex = floor(degreeResult * 4.0f);

			// Increment the bin in the shared histogram
			atomicAdd(&sHist[resultIndex], 2);
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

int main(int argc, char *argv[]) {
	if (argc > 2) {
		std::cerr << "Usage: " << argv[0] << " [DD_device_to_host_ratio]" << std::endl;
		return 1;
	} else if (argc == 2) {
		DD_DEVICE_TO_HOST_RATIO = atof(argv[1]);
	}

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
	
	// Read real data file
	std::ifstream infileD("data_100k_arcmin.txt");

	// Get amount of coordinate pairs
	int nCoordinatePairsD;
	infileD >> nCoordinatePairsD;
	printf("Found %d coordinate pairs in data\n", nCoordinatePairsD);

	// Allocate memory for real data on host
	float *h_D = (float *)calloc(nCoordinatePairsD*2+14, sizeof(float));

	if (h_D == NULL) printf("Allocating memory on host failed");

	int index = 0;

	// Read, convert from arc minutes to degrees and store in D
	double ascD, decD;
	while (infileD >> ascD >> decD) {
		if (index < nCoordinatePairsD * 2) {
			h_D[index++] = (float)(ascD * M_PI / (60.0*180.0));
			h_D[index++] = (float)(decD * M_PI / (60.0*180.0));
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in data\n");
			exit(1);
		}
	}

	float *d_D;
	float *d_R;
	unsigned long long int *h_DD, *h_DR, *h_RR;
	unsigned long long int *d_DD, *d_DR, *d_RR;

	// Allocating and copying the input data to device
	cudaMalloc((void **)&d_D, nCoordinatePairsD * 2 * sizeof(float));

	cudaMemcpy(d_D, h_D, nCoordinatePairsD * 2 * sizeof(float), cudaMemcpyHostToDevice);

	// Allocating the histograms arrays on device
	cudaMalloc((void **)&d_DD, 720 * sizeof(unsigned long long int));
	cudaMalloc((void **)&d_DR, 720 * sizeof(unsigned long long int));
	cudaMalloc((void **)&d_RR, 720 * sizeof(unsigned long long int));

	// Setting all elements of the histograms to zero
	cudaMemset(d_DD, 0, 720 * sizeof(unsigned long long int));
	cudaMemset(d_DR, 0, 720 * sizeof(unsigned long long int));
	cudaMemset(d_RR, 0, 720 * sizeof(unsigned long long int));

	// Device kernel for DD
	int gridSizeX = (nCoordinatePairsD * DD_DEVICE_TO_HOST_RATIO + BLOCKSIZE - 1) / BLOCKSIZE;
	int gridSizeY = (nCoordinatePairsD + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
	dim3 DDGridSize2D(gridSizeX, gridSizeY);
	DD_or_RR_kernel<<<DDGridSize2D, BLOCKSIZE>>>(nCoordinatePairsD * DD_DEVICE_TO_HOST_RATIO, nCoordinatePairsD, d_D, d_DD);

	int *hostComputationResult = (int *)calloc(720, sizeof(int));
	int nCoordinatePairsR;

	#pragma omp parallel
	{
		// Device computation
		#pragma omp single nowait
		{

			// Read synthetic data file
			std::ifstream infileR("flat_100k_arcmin.txt");

			// Get amount of coordinate pairs
			infileR >> nCoordinatePairsR;
			printf("Found %d coordinate pairs in flat\n", nCoordinatePairsR);

			// Allocate memory for synthetic data on host
			float *h_R = (float *)malloc(nCoordinatePairsR * 2 * sizeof(float));

			// Read, convert from arc minutes to degrees and store in R
			double ascR, decR;
			index = 0;
			while (infileR >> ascR >> decR) {
				if (index < nCoordinatePairsR * 2) {
					h_R[index++] = (float)(ascR * M_PI / (60.0*180.0));
					h_R[index++] = (float)(decR * M_PI / (60.0*180.0));
				} else {
					printf("Number of coordinate pairs given in file does not match the actual amount in flat\n");
					exit(1);
				}
			}

			printf("\nRunning host side on %d threads\n", omp_get_num_threads());
			printf("Computing %.2f%% of DD on the host\n\n", (1-DD_DEVICE_TO_HOST_RATIO)*100.0);

			cudaMalloc((void **)&d_R, nCoordinatePairsR * 2 * sizeof(float));
			cudaMemcpy(d_R, h_R, nCoordinatePairsR * 2 * sizeof(float), cudaMemcpyHostToDevice);

			// Device kernel for RR
			gridSizeX = (nCoordinatePairsR + BLOCKSIZE - 1) / BLOCKSIZE;
			gridSizeY = (nCoordinatePairsR + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
			dim3 RRGridSize2D(gridSizeX, gridSizeY);
			DD_or_RR_kernel<<<RRGridSize2D, BLOCKSIZE>>>(nCoordinatePairsR, nCoordinatePairsR, d_R, d_RR);

			// Device kernel for DR
			gridSizeX = (nCoordinatePairsD + BLOCKSIZE - 1) / BLOCKSIZE;
			gridSizeY = (nCoordinatePairsR + ROWSPERTHREAD - 1) / ROWSPERTHREAD;
			dim3 gridSize2D(gridSizeX, gridSizeY);
			DR_kernel<<<gridSize2D, BLOCKSIZE>>>(nCoordinatePairsD, nCoordinatePairsR, d_D, d_R, d_DR);

			// Allocating histograms on host
			h_DD = (unsigned long long int *)malloc(720 * sizeof(unsigned long long int));
			h_DR = (unsigned long long int *)malloc(720 * sizeof(unsigned long long int));
			h_RR = (unsigned long long int *)malloc(720 * sizeof(unsigned long long int));
		}

		// Host computation

		// Each thread has a private histogram and temporary result array
		int *localHist = (int *)calloc(720, sizeof(int));
		float *resultArr = (float *)calloc(nCoordinatePairsD+7, sizeof(float));

		__m256 m1;
		__m256 cosDec1, cosDec2, sinDec1, sinDec2;
		__m256 result;

		__m256 one = _mm256_set1_ps(1.0f);
		__m256 negativeOne = _mm256_set1_ps(-1.0f);	
		// 180/PI*4 from radians to degrees to bin index of size 0.25
		__m256 radToDegToBinIndex = _mm256_set1_ps(229.183118f);

		#pragma omp for schedule(dynamic, 1) nowait
			// Compute 1 - RR_DEVICE_TO_HOST_RATIO of columns in DD on host
			for (int i = nCoordinatePairsD * DD_DEVICE_TO_HOST_RATIO; i < nCoordinatePairsD; i++) {
				__m256 asc1 = _mm256_set1_ps(h_D[i*2]);
				__m256 dec1 = _mm256_set1_ps(h_D[i*2+1]);

				for (int j = i+1; j < nCoordinatePairsD; j +=8) {
					__m256 asc2 = _mm256_loadu_ps(h_D+j*2);
					__m256 dec2 = _mm256_loadu_ps(h_D+j*2+1);

					// sin(dec1) and cos(dec1)
					sincos256_ps(dec1, &sinDec1, &cosDec1);
					// sin(dec2) and cos(dec2)
					sincos256_ps(dec2, &sinDec2, &cosDec2);

					// (asc1-asc2)
					m1 = _mm256_sub_ps(asc1, asc2);
					// cos(asc1-asc2)
					m1 = cos256_ps(m1);				
					// cos(dec2) * cos(asc1-asc2)
					m1 = _mm256_mul_ps(cosDec2, m1);
					// cos(dec1) * cos(dec2) * cos(asc1-asc2)
					m1 = _mm256_mul_ps(cosDec1, m1);

					// sin(dec1) * sin(dec2)
					// tmp = sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(asc1-asc2)
					m1 = _mm256_fmadd_ps(sinDec1, sinDec2, m1);

					// min(tmp, 1)
					m1 = _mm256_min_ps(m1, one);
					// max(tmp, -1)
					m1 = _mm256_max_ps(m1, negativeOne);

					// acos(tmp)
					result = acosv(m1);

					// radian result converted to degrees
					result = _mm256_mul_ps(result, radToDegToBinIndex);

					_mm256_storeu_ps(resultArr+j, result);
				}

				for (int j = i+1; j < nCoordinatePairsD; j++) {
					localHist[(int)resultArr[j]] += 2;
				}
			}

		int total = 0;
		// Each thread updates the global histogram
		for (int i = 0; i < 720; i++) {
			total += localHist[i];
			#pragma omp atomic
				hostComputationResult[i] += localHist[i];
		}

		printf("Total: %d\n", total);
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
	cudaMemcpy(h_DD, d_DD, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_DR, d_DR, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_RR, d_RR, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	// Add the number of galaxies in D to the first bin in DD and respectively in R and RR since we start
	// computing for the index + 1
	h_DD[0] += nCoordinatePairsD;
	h_RR[0] += nCoordinatePairsR;

	// Combine device and host results for RR
	for (int i = 0; i < 720; i++) {
		h_DD[i] += hostComputationResult[i];
	}

	long long totalDR = 0;
	long long totalDD = 0;
	long long totalRR = 0;
	for (int i = 0; i < 720; i++) {
		totalDR += h_DR[i];
		totalDD += h_DD[i];
		totalRR += h_RR[i];
	}

	// Computing the difference
	double result[720];
	for (int i = 0; i < 720; i++) {
		if (h_RR[i] != 0) {
			result[i] = (h_DD[i] - 2.0 * h_DR[i] + h_RR[i]) / (double)h_RR[i];
		} else {
			result[i] = 0;
		}
	}

	printf("Total count in histograms\n");
	printf("DR: %lld\n", totalDR);
	printf("DD: %lld\n", totalDD);
	printf("RR: %lld\n\n", totalRR);

	// Printing the firs t 5 values of the histograms
	printf("DR histogram ");
	for (int i = 0; i < 5; i++) {
		printf("%llu ", h_DR[i]);
	}
	printf("\nDD histogram ");
	for (int i = 0; i < 5; i++) {
		printf("%llu ", h_DD[i]);
	}
	printf("\nRR histogram ");
	for (int i = 0; i < 5; i++) {
		printf("%llu ", h_RR[i]);
	}
	printf("\nOmega values ");
	for (int i = 0; i < 5; i++) {
		printf("%lf ", result[i]);
	}
	printf("\n");
}
