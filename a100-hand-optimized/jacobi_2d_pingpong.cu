#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "measure.h"

// The problem shape. These are the values ../tt-source/jacobi_2d.cu is built
// with -- agreeing on them is the point, not sharing a file. PolyBench spells
// them N / _PB_N / _PB_TSTEPS and the verbatim kernel below uses those names,
// so they stay macros.
#define N 3200          // 10*32*10
#define _PB_N 3200
#define _PB_TSTEPS 1000
#define DATA_TYPE float

// ---------------------------------------------------------------------------
// Measured configuration. Fixed on purpose: --device and --warmup-ms are the
// only flags, and there is no -D.
//
// ---------------------------------------------------------------------------
namespace cfg {
constexpr int kTsteps = _PB_TSTEPS;  // 1000
constexpr int kWgX = 128;            // measured, not inherited
constexpr int kWgY = 2;
constexpr unsigned kSeed = 42;       // the field's only input; fixes the checksum
}  // namespace cfg

// ---------------------------------------------------------------------------
// VERBATIM from reference-cuda/tt-source/jacobi_2d.cu.
// ---------------------------------------------------------------------------

__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		B[i*N + j] = 0.2f * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (1 + j)] + A[(1 + i)*N + j] + A[(i-1)*N + j]);
	}
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  // 5 and 400 ms for the same reasons as ../a100-measure; see measure.h.
  for (int i = 1; i < argc; i++) {
    if (parse_warmup_flag(argc, argv, &i)) continue;
    else if (!strcmp(argv[i], "--device") && i + 1 < argc)
      CUDA_CHECK(cudaSetDevice(atoi(argv[++i])));
    else {
      fprintf(stderr,
              "usage: %s [--device D] [--warmup-ms W]\n"
              "  Everything else is fixed: N=%d, %d timesteps, %dx%d work group,\n"
              "  seed %u, ping-pong, no per-launch sync. See the cfg:: note in\n"
              "  the source for why there are no knobs.\n",
              argv[0], N, cfg::kTsteps, cfg::kWgX, cfg::kWgY, cfg::kSeed);
      return 1;
    }
  }

  const size_t total = (size_t)N * N;
  const size_t bytes = total * sizeof(DATA_TYPE);

  DATA_TYPE *hA = (DATA_TYPE *)malloc(bytes);
  DATA_TYPE *hOut = (DATA_TYPE *)malloc(bytes);
  if (!hA || !hOut) { fprintf(stderr, "host alloc failed\n"); return 1; }
  fill_uniform(hA, total, cfg::kSeed);

  DATA_TYPE *dA = NULL, *dB = NULL;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));

  // Launch config: VERBATIM from tt-source runJacobi2DCUDA().
  dim3 block(cfg::kWgX, cfg::kWgY);
  dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), (unsigned int)ceil( ((float)N) / ((float)block.y) ));

  // Both buffers start from the same field — see the boundary note at the top.
  auto upload = [&]() {
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hA, bytes, cudaMemcpyHostToDevice));
  };

  // Stencil src->dst, then swap. Returns whichever buffer was written last,
  // which the caller needs for the D2H.
  auto run_steps = [&](int steps) {
    DATA_TYPE *src = dA, *dst = dB;
    for (int s = 0; s < steps; s++) {
      runJacobiCUDA_kernel1<<<grid, block>>>(N, src, dst);
      DATA_TYPE *tmp = src; src = dst; dst = tmp;
    }
    return src;  // post-swap, src holds the last write
  };

  upload();
  // --warmup-ms 0 means no warm-up AT ALL -- warmup()'s loop is a do/while and
  // would still launch once. dram_traffic.py passes it so that the launches
  // Nsight Compute sees are the timed span and nothing else.
  if (g_warmup_ms > 0) warmup([&]() { run_steps(1); });
  check_launch("jacobi warmup");

  Timers t;
  timers_init(&t);
  ClockWatch clk;
  clockwatch_init(&clk);

  // (a) Kernel-only. The stencil's work count is value-independent, so no
  // re-init between reps is needed.
  upload();
  CUDA_CHECK(cudaDeviceSynchronize());
  // ONE timed span, then the process ends: one process is one measurement, the
  // level the Tenstorrent side's board reset separates its measurements at.
  // Dispersion is bench.py's, across the fresh processes it runs.
  timer_start(&t);
  DATA_TYPE *last = run_steps(cfg::kTsteps);
  float kernel_ms = timer_stop_ms(&t);
  clockwatch_sample(&clk);
  check_launch("jacobi kernel-only");

  // Checksum of the timed span, one untimed D2H. `last` is whichever
  // buffer the final swap left the field in.
  CUDA_CHECK(cudaMemcpy(hOut, last, bytes, cudaMemcpyDeviceToHost));

  double mean, stdv;
  summarize(hOut, total, &mean, &stdv);

  char shape[80];
  snprintf(shape, sizeof(shape), "N=%d,tsteps=%d,wg=%dx%d,no-sync", N, cfg::kTsteps,
           cfg::kWgX, cfg::kWgY);
  // 1 launch per timestep under ping-pong (the baseline needs 2).
  emit_json("jacobi_2d_pingpong", shape, cfg::kSeed, cfg::kTsteps, kernel_ms, &clk,
            mean, stdv);
  fprintf(stderr,
          "jacobi_2d_pingpong: %.3f us/timestep (kernel-only), "
          "SM %u-%u MHz of %u\n",
          kernel_ms * 1000.0 / cfg::kTsteps, clk.min_mhz, clk.max_mhz, clk.boost_mhz);

  timers_free(&t);
  cudaFree(dA);
  cudaFree(dB);
  free(hA);
  free(hOut);
  return 0;
}
