#include <stdio.h> 	/* printf */
#include <fstream> 	/* ifstream */
#include <math.h> 	/* M_PI */
#include <stdlib.h>	/* malloc, calloc, exit */
#include <immintrin.h> /* AVX */

int main(void) {
	bool AVX = true;

	// Read real data file
	std::ifstream infileD("data_100k_arcmin.txt");

	// Get amount of coordinate pairs
	int nCoordinatePairsD;
	infileD >> nCoordinatePairsD;
	printf("Found %d coordinate pairs in data\n", nCoordinatePairsD);

	// Allocate memory for real data on host
	float *h_ascD = (float *)aligned_alloc(32, nCoordinatePairsD * sizeof(float));
	float *h_decD = (float *)aligned_alloc(32, nCoordinatePairsD * sizeof(float));


	if (h_ascD == NULL || h_decD == NULL) printf("Allocating memory on host failed");

	int index = 0;

	// Read, convert from arc minutes to degrees and store in D
	double ascD, decD;
	while (infileD >> ascD >> decD) {
		if (index < nCoordinatePairsD) {
			h_ascD[index] = (float)(ascD * M_PI / (60.0*180.0));
			h_decD[index++] = (float)(decD * M_PI / (60.0*180.0));
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in data\n");
			exit(1);
		}
	}

	// Allocating result array
	unsigned long long int *hist = (unsigned long long int *)calloc(720, sizeof(unsigned long long int));

	if (AVX) {
		/* Vectorized part */
		float *result = (float *)aligned_alloc(32, nCoordinatePairsD * sizeof(float));

		__m256 m1, m2, m3, m4, m5, m6;
		__m256 one = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
		__m256 negativeOne = _mm256_set_ps(-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);

		__m256 *pAsc1 = (__m256 *)h_ascD;
		__m256 *pAsc2 = (__m256 *)h_ascD;
		__m256 *pDec1 = (__m256 *)h_decD;
		__m256 *pDec2 = (__m256 *)h_decD;
		__m256 *pResult;

		for (int i = 0; i < 40000; i++) {
			// Reset result pointer to start of result array
			pResult = (__m256 *)result;

			for (int j = 0; j < 40000 / 8; j++) {

				// (asc1-asc2)
				m1 = _mm256_sub_ps(*pAsc1, *pAsc2);
				// cos(asc1-asc2)
				m1 = _mm256_cos_ps(m1);
				// cos(dec2)
				m2 = _mm256_cos_ps(*pDec2);
				// cos(dec2) * cos(asc1-asc2)
				m1 = _mm256_mul_ps(m2, m1);
				// cos(dec1)
				m2 = _mm256_cos_ps(*pDec1);
				// cos(dec1) * cos(dec2) * cos(asc1-asc2)
				m1 = _mm256_mul_ps(m2, m1);
				// sin(dec1)
				m2 = _mm256_sin_ps(*pDec1);
				// sin(dec2)
				m3 = _mm256_sin_ps(*pDec2);
				// sin(dec1) * sin(dec2)
				m2 = _mm256_mul_ps(m2, m3);
				// tmp = sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(asc1-asc2)
				m1 = _mm256_add_ps(m1, m2);

				// min(tmp, 1)
				m1 = _mm256_min_ps(m1, one);
				// max(tmp, -1)
				m1 = _mm256_max_ps(m1, negativeOne);

				// acos(tmp)
				*pResult = _mm256_acos_ps(m1);

				pAsc2++;
				pDec2++;
				pResult++;
			}

			for (int j = 0; j < 40000; j++) {
				float degreeResult = result[j] * 180.0f/3.14159f;
				int resultIndex = floor(degreeResult * 4.0f);
				hist[resultIndex]++;
			}

			pAsc1++;
			pDec1++;
		}
	} else {
		for (int i = 0; i < 40000; i++) {

			float asc1 = h_ascD[i];
			float dec1 = h_decD[i];

			for (int j = 0; j < 40000; j++) {

				float asc2 = h_ascD[j];
				float dec2 = h_decD[j];

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
				hist[resultIndex]++;
			}
		}
	}
}