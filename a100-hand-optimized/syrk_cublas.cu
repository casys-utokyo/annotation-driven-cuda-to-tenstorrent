#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "measure.h"

// The problem shape and ../tt-source/syrk.cu's constants (it #defines the
// latter inside the kernel body). Agreeing on the values is the point.
#define NI 3200         // 10*32*10
#define NJ 3200
#define ALPHA 2.0f
#define BETA 3.0f

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t _s = (expr);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "%s:%d: cuBLAS error %d (%s)\n", __FILE__, __LINE__,     \
              (int)_s, #expr);                                                 \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef __nv_bfloat16 bf16;

// ---------------------------------------------------------------------------
// Measured configuration. Fixed on purpose: --device and --warmup-ms are the
// only flags.
//
// kInner=100 matches the TT build's 100 iterations inside one generic, and
// makes the span long enough to pass the point where the 250 W cap pulls the
// SM clock off boost, so it is the sustained number rather than the figure a
// single isolated launch gives.
// ---------------------------------------------------------------------------
namespace cfg {
constexpr int kInner = 100;
constexpr unsigned kSeed = 42;  // A from kSeed, C from kSeed+1; fixes the checksum
}  // namespace cfg

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (parse_warmup_flag(argc, argv, &i)) continue;
    else if (!strcmp(argv[i], "--device") && i + 1 < argc)
      CUDA_CHECK(cudaSetDevice(atoi(argv[++i])));
    else {
      fprintf(stderr,
              "usage: %s [--device D] [--warmup-ms W]\n"
              "  Everything else is fixed: NI=NJ=%d, alpha=%.1f beta=%.1f, bf16\n"
              "  in / fp32 accumulate, %d launches per span. --warmup-ms 0 also\n"
              "  skips the checksum, leaving the run exactly the span. See the\n"
              "  in the source. Do NOT pass --warmup-ms 0 for a quoted number:\n"
              "  the span then reports +53%%, the clock never leaving idle.\n",
              argv[0], NI, ALPHA, BETA, cfg::kInner);
      return 1;
    }
  }

  const size_t a_total = (size_t)NI * NJ, c_total = (size_t)NI * NI;
  const size_t a_bytes = a_total * sizeof(bf16), c_bytes = c_total * sizeof(bf16);

  // fp32 host buffers from measure.h's generator with seed / seed+1 — the same
  // inputs every driver in this directory uses — staged down to bf16 once, the
  // same rounding the TT side applies. hC stays fp32 for the checksum.
  float *hA = (float *)malloc(a_total * sizeof(float));
  float *hC = (float *)malloc(c_total * sizeof(float));
  bf16 *sA = (bf16 *)malloc(a_bytes);
  bf16 *sC = (bf16 *)malloc(c_bytes);
  // D2H lands HERE, never back into sC. Reusing the upload buffer as the
  // download buffer would feed rep r's result into rep r+1 as input, and with
  // beta=3 that compounds 3x per rep — the timing is unaffected but the
  // checksum stops being "the output of one syrk".
  bf16 *hOut = (bf16 *)malloc(c_bytes);
  if (!hA || !hC || !sA || !sC || !hOut) {
    fprintf(stderr, "host alloc failed\n"); return 1;
  }
  fill_uniform(hA, a_total, cfg::kSeed);
  fill_uniform(hC, c_total, cfg::kSeed + 1);
  for (size_t i = 0; i < a_total; i++) sA[i] = __float2bfloat16(hA[i]);
  for (size_t i = 0; i < c_total; i++) sC[i] = __float2bfloat16(hC[i]);

  bf16 *dA = NULL, *dC = NULL;
  CUDA_CHECK(cudaMalloc(&dA, a_bytes));
  CUDA_CHECK(cudaMalloc(&dC, c_bytes));

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  const float alpha = ALPHA, beta = BETA;

  // C is re-uploaded before every launch because beta*C compounds 3x per launch
  // and would otherwise overflow to inf. A never changes.
  CUDA_CHECK(cudaMemcpy(dA, sA, a_bytes, cudaMemcpyHostToDevice));
  auto upload = [&]() {
    CUDA_CHECK(cudaMemcpy(dC, sC, c_bytes, cudaMemcpyHostToDevice));
  };
  auto run_syrk = [&]() {
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, NI, NI, NJ, &alpha,
                              dA, CUDA_R_16BF, NJ, dA, CUDA_R_16BF, NJ, &beta,
                              dC, CUDA_R_16BF, NI, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };

  upload();
  // --warmup-ms 0 means no warm-up AT ALL -- warmup()'s loop is a do/while and
  // would still launch once. dram_traffic.py passes it so that the launches
  // Nsight Compute sees are the timed span and nothing else.
  // The checksum needs its OWN single syrk: C is read-modify-written with
  // beta=3, so the span's `inner` launches compound it and in bf16 it saturates
  // to inf past ~80. It is taken here, with the warm-up, because a run with
  // --warmup-ms 0 is a profiling run -- dram_traffic.py refuses any other kind
  // -- and there the extra launch would be the only thing standing between the
  // profiled launches and the timed span. So with a warm-up this behaves like
  // the other two drivers; without one, the run is exactly the span.
  double mean = 0, stdv = 0;
  if (g_warmup_ms > 0) {
    warmup(run_syrk);
    upload();
    run_syrk();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOut, dC, c_bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < c_total; i++) hC[i] = __bfloat162float(hOut[i]);
    summarize(hC, c_total, &mean, &stdv);  // expect 0.170051 / 13.361197
  }
  check_launch("syrk warmup");

  Timers t;
  timers_init(&t);
  ClockWatch clk;
  clockwatch_init(&clk);

  // Kernel-only, ONE timed span of `inner` back-to-back launches: one process
  // is one measurement, the level the Tenstorrent side's board reset separates
  // its measurements at. Dispersion is bench.py's, across the fresh processes
  // it runs. The re-upload sits outside the timed region.
  upload();
  CUDA_CHECK(cudaDeviceSynchronize());
  timer_start(&t);
  for (int k = 0; k < cfg::kInner; k++) run_syrk();
  float kernel_ms = timer_stop_ms(&t);
  clockwatch_sample(&clk);
  check_launch("syrk kernel-only");

  char shape[64];
  snprintf(shape, sizeof(shape), "NI=%d,NJ=%d,bf16,inner=%d", NI, NJ, cfg::kInner);
  // `kernel_ms` is the span and `launches_per_rep` is what happened in it, the
  // schema the other two drivers use; `kernel_us_per_launch` is the per-SYRK
  // figure to quote.
  emit_json("syrk_cublas", shape, cfg::kSeed, cfg::kInner, kernel_ms, &clk, mean, stdv);
  // Useful-symmetric-FLOP for C = alpha*A*A' + beta*C.
  double gflop = 2.0 * (double)NI * NI * NJ / 1e9;
  double per_syrk = kernel_ms / cfg::kInner;
  fprintf(stderr,
          "syrk_cublas(bf16): %.3f ms per syrk (%d back-to-back in the span), "
          "%.1f GFLOP/s, SM %u-%u MHz of %u\n",
          per_syrk, cfg::kInner, gflop / (per_syrk / 1e3), clk.min_mhz, clk.max_mhz,
          clk.boost_mhz);

  timers_free(&t);
  cublasDestroy(h);
  cudaFree(dA);
  cudaFree(dC);
  free(hA);
  free(hC);
  free(sA);
  free(sC);
  free(hOut);
  return 0;
}
