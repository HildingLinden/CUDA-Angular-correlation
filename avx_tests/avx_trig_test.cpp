#include <immintrin.h>  /* AVX */
#include <stdio.h>      /* printf */
#include <math.h>
#include "avx_mathfun.h"

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

void differenceTest() {
    float testValue = 0.000f;
    float maxDiffAcos = 0.0f, maxDiffCos = 0.0f, maxDiffSin = 0.0f;

    for (int i = 0; i < 10001; i++) {
        __m256 x = _mm256_set1_ps(testValue);
        __m256 result1, result2;
        float *vecResult1 = (float *)&result1;
        float *vecResult2 = (float *)&result2;

        // Acos
        result1 = acosv(x);        
        float mathResult = acosf(testValue);

        float difference = vecResult1[0]-mathResult;
        maxDiffAcos = fmaxf(maxDiffAcos, difference);

        // Sin and Cos
        sincos256_ps(x, &result1, &result2);
        mathResult = cosf(testValue);

        difference = vecResult2[0]-mathResult;
        maxDiffCos = fmaxf(maxDiffCos, difference);

        mathResult = sinf(testValue);

        difference = vecResult1[0]-mathResult;
        maxDiffSin = fmaxf(maxDiffSin, difference);

        testValue += 0.0001f;
    }

    printf("Max difference for values 0-1000\n");
    printf("Acos: %f, Cos: %f, Sin: %f\n", maxDiffAcos, maxDiffCos, maxDiffSin);
}

void performanceTestMath() {
    float testValue = 0.000f;
    float result1, result2, result3;
    for (int i = 0; i < 800000; i++) {
        result1 = acosf(testValue);
        result2 = cosf(testValue);
        result3 = sinf(testValue);

        testValue += 0.0000001f;
    }
}

void performanceTestVec() {
    float testValue = 0.000f;
    __m256 result1, result2, result3;
    for (int i = 0; i < 800000/4; i++) {
        __m256 x = _mm256_set1_ps(testValue);

        result1 = acosv(x);
        sincos256_ps(x, &result2, &result3);

        testValue += 0.0000008f;
    }
}

int main(void) {
    //performanceTestMath();
    performanceTestVec();

    return 0;
}