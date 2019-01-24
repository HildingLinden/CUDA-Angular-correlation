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
Read files containing coordinates
Allocate memory for coordinates on host
Read in coordinates, convert them from arc minutes to radians and store them in memory
? Print information about Host/device ?
Allocate memory for the coordinates on the device
Copy the coordinates to the device
Allocate zero-initialized memory on host for the histograms
Allocate memory on device for the histograms
Copy the zero-initialized memory from host to device
Compute the grid size and block size for the kernels
Launch kernels for D * R, D * D and R * R

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