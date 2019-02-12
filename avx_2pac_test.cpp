#include <stdio.h> 			/* printf */
#include <fstream> 			/* ifstream */
#include <math.h> 			/* M_PI */
#include <cmath>			/* lround */
#include <stdlib.h>			/* malloc, calloc, exit */
#include <immintrin.h> 		/* AVX */
#include "avx_mathfun.h" 	/* sincos256_ps */

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

int main(void) {
	bool AVX = true;

	// Read real data file
	std::ifstream infileD("data_100k_arcmin.txt");

	// Get amount of coordinate pairs
	int nCoordinatePairsD;
	infileD >> nCoordinatePairsD;
	printf("Found %d coordinate pairs in data\n", nCoordinatePairsD);

	// Allocate memory for real data on host
	float *h_ascD = (float *)calloc(nCoordinatePairsD+7, sizeof(float));
	float *h_decD = (float *)calloc(nCoordinatePairsD+7, sizeof(float));


	if (h_ascD == NULL || h_decD == NULL) printf("Allocating memory on host failed");

	int index = 0;

	// Read, convert from arc minutes to degrees and store in D
	double ascD, decD;
	while (infileD >> ascD >> decD) {
		if (index < nCoordinatePairsD) {
			h_ascD[index] = (float)(ascD * M_PI / (60.0*180.0));
			h_decD[index] = (float)(decD * M_PI / (60.0*180.0));
			index++;
		} else {
			printf("Number of coordinate pairs given in file does not match the actual amount in data\n");
			exit(1);
		}
	}

	// Allocating result array
	unsigned long long int *hist = (unsigned long long int *)calloc(720, sizeof(unsigned long long int));

	if (AVX) {
		/* Vectorized part */
		float *resultArr = (float *)calloc(nCoordinatePairsD+7, sizeof(float));

		__m256 m1;
		__m256 cosDec1, cosDec2, sinDec1, sinDec2;
		__m256 result;

		__m256 one = _mm256_set1_ps(1.0f);
		__m256 negativeOne = _mm256_set1_ps(-1.0f);	
		// 180/PI for rad to deg and 180/PI*4 to bin index
		__m256 radToDegToBinIndex = _mm256_set1_ps(229.183118f);

		for (int i = 0; i < 100000; i++) {
			__m256 asc1 = _mm256_set1_ps(h_ascD[i]);
			__m256 dec1 = _mm256_set1_ps(h_decD[i]);

			for (int j = 0; j < (100000+7)/8; j++) {
				__m256 asc2 = _mm256_loadu_ps(h_ascD+j*8);
				__m256 dec2 = _mm256_loadu_ps(h_decD+j*8);

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

				// floor(result)
				result = _mm256_floor_ps(result);

				// integer mask

				_mm256_storeu_ps(resultArr+j*8, result);
			}

			for (int j = 0; j < 100000; j++) {
				hist[(int)resultArr[j]] += 1;
			}
		}

		//hist[0] += 100000;
	} else {
		for (int i = 0; i < 20000; i++) {

			float asc1 = h_ascD[i];
			float dec1 = h_decD[i];

			for (int j = i+1; j < 20000; j++) {

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
				hist[resultIndex] += 2;
			}
		}
	}

	unsigned long long int sum = 0;
	for (int i = 0; i < 720; i++) {
		sum += hist[i];
	}
	printf("Sum of histogram: %llu\n", sum);

	for (int i = 0; i < 5; i++) {
		printf("%llu ", hist[i]);
	}
	printf("\n");
}