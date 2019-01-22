# CUDA-Angular-correlation

The shared memory histogram is of type unsigned integer instead of unsigned long long integer which the global histogram is. This is possible because in the worst case scenario a bin can not have a value over blocksize*rowsperblock. This brings down the register count from 40 to 32 which improves the occupancy and the atomicAdd might also be faster with the smaller type.

With 32 registers per thread block, the blocksize is optimal at 128, 256, 512 or 1024. The smaller the blocksize the less threads are idle on the edge of the domain. But other warps might be able to run.

Rows per thread has similar performance above 32.

--use_fast_math force some functions/operators to use an instrinsic version. See CUDA programming guide Appendix E.2.
Maximum errorfor -PI < x < PI:
__sinf(x): 2^-21.41
__cosf(x): 2^-21.19

32-bit floating-point sine & cosine results per clock cycle on GPUs with compute capability 6.1: 32