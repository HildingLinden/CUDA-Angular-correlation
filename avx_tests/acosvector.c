#include <immintrin.h>      /* AVX */

static inline __m128 acosf4(__m128 x)
{
    __m128 xabs = fabsf4(x);
	__m128 select = _mm_cmplt_ps( x, _mm_setzero_ps() );
    __m128 t1 = sqrtf4(vec_sub(_mm_set1_ps(1.0f), xabs));
    
    /* Instruction counts can be reduced if the polynomial was
     * computed entirely from nested (dependent) fma's. However, 
     * to reduce the number of pipeline stalls, the polygon is evaluated 
     * in two halves (hi amd lo). 
     */
    __m128 xabs2 = _mm_mul_ps(xabs,  xabs);
    __m128 xabs4 = _mm_mul_ps(xabs2, xabs2);
    __m128 hi = vec_madd(vec_madd(vec_madd(_mm_set1_ps(-0.0012624911f),
		xabs, _mm_set1_ps(0.0066700901f)),
			xabs, _mm_set1_ps(-0.0170881256f)),
				xabs, _mm_set1_ps( 0.0308918810f));
    __m128 lo = vec_madd(vec_madd(vec_madd(_mm_set1_ps(-0.0501743046f),
		xabs, _mm_set1_ps(0.0889789874f)),
			xabs, _mm_set1_ps(-0.2145988016f)),
				xabs, _mm_set1_ps( 1.5707963050f));
    
    __m128 result = vec_madd(hi, xabs4, lo);
    
    // Adjust the result if x is negactive.
    return vec_sel(
		vec_mul(t1, result),									// Positive
		vec_nmsub(t1, result, _mm_set1_ps(3.1415926535898f)),	// Negative
		select);
}

