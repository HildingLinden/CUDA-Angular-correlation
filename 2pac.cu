#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

int main(void) {
	// Context creation
	//cudaFree(0);

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		printf("Device nr.: %d\n", i);
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