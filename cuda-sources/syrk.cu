/**
 * syrk.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "syrk.cuh"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

#define alpha 2.0f
#define beta 3.0f

__global__ void syrk_kernel(int ni, int nj, DATA_TYPE *a, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NI))
	{
		int k;
		float acc = 0.0;	
		for(k=0; k < _PB_NJ; k++)
		{
			acc += a[i * NJ + k] * a[j * NJ + k];
		}
		c[i * NI + j] = beta * c[i * NI + j] + alpha * acc;
	}
}

void syrkCuda(DATA_TYPE* A_gpu, DATA_TYPE* C_gpu, int ni, int nj)
{
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_X))), (size_t)ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_Y)));

	/* Start timer. */
  	// polybench_start_instruments;

	syrk_kernel<<<grid,block>>>(ni, nj, A_gpu,C_gpu);
	cudaThreadSynchronize();

}
