#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}

int main(void) {
	//context creation
	cudaFree(0);

	int n = 1 << 30;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(n*sizeof(float));
	y = (float*)malloc(n*sizeof(float));

	cudaMalloc(&d_x, n*sizeof(float));
	cudaMalloc(&d_y, n*sizeof(float));

	for (int i = 0; i < n; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

	saxpy<<<(n+255)/256, 256>>>(n, 2.0f, d_x, d_y);

	cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < n; ++i) {
		maxError = max(maxError, abs(y[i]-4.0f));
	}
	printf("Max error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}