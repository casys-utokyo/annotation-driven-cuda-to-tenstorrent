/**
 * jacobi2D.cuh: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef JACOBI2D_CUH
# define JACOBI2D_CUH

// /* Default to STANDARD_DATASET. */

// /* Do not define anything if the user manually defines the size. */
// /* Define the possible dataset sizes. */

#define N 10*32*10

# define _PB_TSTEPS 1000
# define _PB_N 10*32*10

# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

/* Thread block dimensions */

#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32

#endif /* !JACOBI2D*/