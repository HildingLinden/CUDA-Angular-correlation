# CUDA-Angular-correlation

The shared memory histogram is of type unsigned integer instead of unsigned long long integer which the global histogram is. This is possible because in the worst case scenario a bin can not have a value over blocksize*rowsperblock. This brings down the register count from 40 to 32 which improves the occupancy and the atomicAdd might also be faster with the smaller type.

With 32 registers per thread block, the blocksize is optimal at 128, 256, 512 or 1024. The smaller the blocksize the less threads are idle on the edge of the domain. But other warps might be able to run.

Rows per thread has similar performance above 32.

--use_fast_math force some functions/operators to use an instrinsic version. See CUDA programming guide Appendix E.2.
Maximum errorfor -PI < x < PI:
__sinf(x): 2^-21.41
__cosf(x): 2^-21.19

32-bit floating-point sine & cosine results per clock cycle on GPUs with compute capability 6.1: 32

Overall structure:  
/* Host */
Print information about Host/device  
Read files
	Read number of coordinate pairs
	Allocate memory for coordinates on host  
	Read in coordinates, convert them from arc minutes to radians and store them in memory  
Allocate memory for the coordinates on the device  
Copy the coordinates to the device    
Allocate memory on device for the histograms
Fill device histogram memory with zeroes
Compute the grid size and block size for the kernels  
Launch kernels

The steps are fist done for the Coordinates in D so that the computation of DD can start as soon as possible  

Allocate memory for histograms on host
Copy the histograms from device to host when the computation is done
Compute the omega values for the histograms
Print the results
  
/* Device */  
Compute the x and y coordinate (Effectively the index in the first and second set of galaxies)  
Check that the current x is inside the domain  
Declare a histogram in shared memory  
Thread zero initialize the histogram with zeroes  
Synchronize the threads  
Get the right ascension and declination of the first set  
Get the right ascension and declination of the second set  
Check that the difference between the two coordinate pairs are large than some epsilon  
Compute the distance between the pairs in radians using the formula **arccos(sin(dec<sub>1</sub>) * sin(dec<sub>2</sub>) + cos(dec<sub>1</sub>) * cos(dec<sub>2</sub>) * cos(asc<sub>1</sub>-asc<sub>2</sub>))**  
Convert the distance to degrees  
Compute the histogram bin  
Increment the local histogram bin  
Every thread does this for a column and ROWSPERTHREAD rows or until the domain edge  
Synchronize threads after all computation is done  
Thread zero update the global histogram with the local histogram  
  
/* Host */  
Check for synchronous and asynchronous errors on the device  
Copy the histograms to the host  
Increment the first bin of DD and RR for every element in D and R, respectively.  
Calculate the total number of results in the histograms, they should add up to 10 billion.  
Calculate the omega value of the bins in the histograms using the formula **(DD<sub>i</sub> - 2 * DR<sub>i</sub> + RR<sub>i</sub>) / RR<sub>i</sub>**  
Free memory on host and device (This is not strictly necessary)  
  
AVX  
Using SVML with ICC for sin, cos & acos: http://climserv.ipsl.polytechnique.fr/documentation/intel-icc/Getting_Started.htm  
Using avx_mathfun with gcc for sin & cos: http://software-lisc.fbk.eu/avx_mathfun/  
https://github.com/reyoung/avx_mathfun/blob/master/avx_mathfun.h  
acos for positive values: https://stackoverflow.com/questions/46974513/code-for-acos-with-avx256  
General notes  
http://web.archive.org/web/20150531202539/http://www.codeproject.com/Articles/4522/Introduction-to-SSE-Programming  
https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX  
https://stackoverflow.com/a/36387954.  
Possible mask
int diff = 100000-j*8;
__m256i mask = _mm256_setr_epi32(-1, (diff-1)*(-1), (diff-2)*(-1), (diff-3)*(-1), (diff-4)*(-1), (diff-5)*(-1), (diff-6)*(-1), (diff-7)*(-1));  