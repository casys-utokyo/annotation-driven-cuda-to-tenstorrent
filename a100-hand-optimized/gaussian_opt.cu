#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "measure.h"  // local copy; this directory is self-contained


// Verbatim from upstream gpu-rodinia cuda/gaussian/gaussian.cu.
static void create_matrix(float *m, int size) {
  int i, j;
  float lamda = -0.01;
  float *coe = (float *)malloc((2 * size - 1) * sizeof(float));
  float coe_i = 0.0;

  for (i = 0; i < size; i++) {
    coe_i = 10 * exp(lamda * i);
    j = size - 1 + i;
    coe[j] = coe_i;
    j = size - 1 - i;
    coe[j] = coe_i;
  }
  for (i = 0; i < size; i++)
    for (j = 0; j < size; j++) m[i * size + j] = coe[size - 1 - i + j];
  free(coe);
}

// Fan1 is verbatim upstream: it is O(Size) work and was never the problem.
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t)
{
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

// Coalesced Fan2: identical arithmetic to upstream's, threadIdx.x now walks the
// column so a warp's 32 lanes cover 32 consecutive floats of one row.
__global__ void Fan2_coal(float *m, float *a, int Size, int t)
{
  int col = t + blockIdx.x * blockDim.x + threadIdx.x;
  int row = t + 1 + blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= Size || row >= Size) return;

  float mult = m[Size * row + t];
  a[Size * row + col] -= mult * a[Size * t + col];
}

// ---------------------------------------------------------------------------
// Host: the elimination loop.
// ---------------------------------------------------------------------------

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Measured configuration. Fixed on purpose: --device and --warmup-ms are the
// only flags.
//
// kFan2Bx=256 gives a warp eight full 128 B transactions of one row while
// keeping the block at 256 threads; kFan1Block=128 gives Fan1 a grid the
// scheduler can spread.
//
// No seed: create_matrix is upstream Rodinia's deterministic Toeplitz.
// ---------------------------------------------------------------------------
namespace cfg {
constexpr int kSize = 16 * 32 * 10;  // 5120; tt-source/gaussian.cu's shape
constexpr int kFan1Block = 128;
constexpr int kFan2Bx = 256;
constexpr int kFan2By = 1;
}  // namespace cfg

static void ForwardSub(float *dM, float *dA) {
  constexpr int Size = cfg::kSize;
  for (int t = 0; t < Size - 1; t++) {
    int rows = Size - 1 - t;  // rows still to eliminate
    int cols = Size - t;      // columns still live (t .. Size-1)

    // Unconditional, as upstream is: `rows` >= 1 and `cols` >= 2 for every t in
    // [0, Size-1), so every ceil_div here is >= 1 and a guard on it would be
    // dead code.
    Fan1<<<ceil_div(rows, cfg::kFan1Block), cfg::kFan1Block>>>(dM, dA, Size, t);
    Fan2_coal<<<dim3(ceil_div(cols, cfg::kFan2Bx), ceil_div(rows, cfg::kFan2By)),
                dim3(cfg::kFan2Bx, cfg::kFan2By)>>>(dM, dA, Size, t);
  }
}

int main(int argc, char **argv) {
  constexpr int Size = cfg::kSize;

  for (int i = 1; i < argc; i++) {
    if (parse_warmup_flag(argc, argv, &i)) continue;
    else if (!strcmp(argv[i], "--device") && i + 1 < argc)
      CUDA_CHECK(cudaSetDevice(atoi(argv[++i])));
    else {
      fprintf(stderr,
              "usage: %s [--device D] [--warmup-ms W]\n"
              "  Everything else is fixed: Size=%d, Fan1 block %d, Fan2 block\n"
              "  %dx%d, coalesced Fan2, no per-launch sync. See the cfg:: note\n"
              "  in the source for why there are no knobs.\n",
              argv[0], Size, cfg::kFan1Block, cfg::kFan2Bx, cfg::kFan2By);
      return 1;
    }
  }

  const size_t total = (size_t)Size * Size;
  const size_t bytes = total * sizeof(float);

  float *hA = (float *)malloc(bytes);
  float *hM = (float *)calloc(total, sizeof(float));
  float *hOut = (float *)malloc(bytes);
  if (!hA || !hM || !hOut) {
    fprintf(stderr, "host alloc failed\n"); return 1;
  }
  // upstream's own create_matrix, a symmetric exp-decay Toeplitz. GE's work is
  // value-independent, so the input only decides the checksum -- and it has to
  // be this one, because that is what ../a100-measure's baseline uses and the
  // checksums are compared.
  create_matrix(hA, Size);

  float *dA = NULL, *dM = NULL;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dM, bytes));

  auto upload = [&]() {
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dM, hM, bytes, cudaMemcpyHostToDevice));
  };

  // Warm up (untimed) on t=0 only — a full ForwardSub costs seconds at the
  // default size. It scribbles over dA/dM/dB; every timed rep re-uploads.
  upload();
  // --warmup-ms 0 means no warm-up AT ALL -- warmup()'s loop is a do/while and
  // would still launch once. dram_traffic.py passes it so that the launches
  // Nsight Compute sees are the timed span and nothing else.
  if (g_warmup_ms > 0) warmup([&]() {
    int rows = Size - 1, cols = Size;
    Fan1<<<ceil_div(rows, cfg::kFan1Block), cfg::kFan1Block>>>(dM, dA, Size, 0);
    Fan2_coal<<<dim3(ceil_div(cols, cfg::kFan2Bx), ceil_div(rows, cfg::kFan2By)),
                dim3(cfg::kFan2Bx, cfg::kFan2By)>>>(dM, dA, Size, 0);
  });
  check_launch("gaussian warmup");

  Timers t;
  timers_init(&t);
  ClockWatch clk;
  clockwatch_init(&clk);

  // Kernel-only, ONE timed span: one process is one measurement, the level the
  // Tenstorrent side's board reset separates its measurements at. Dispersion
  // is bench.py's, across the fresh processes it runs. The upload sits outside
  // the timed region (ForwardSub eliminates in place).
  upload();
  CUDA_CHECK(cudaDeviceSynchronize());
  timer_start(&t);
  ForwardSub(dM, dA);
  float kernel_ms = timer_stop_ms(&t);
  clockwatch_sample(&clk);
  check_launch("gaussian kernel-only");

  // Checksum of the timed span, one untimed D2H: ForwardSub eliminates in
  // place, so the result is in dA.
  CUDA_CHECK(cudaMemcpy(hOut, dA, bytes, cudaMemcpyDeviceToHost));

  double mean, stdv;
  summarize(hOut, total, &mean, &stdv);

  char shape[128];
  snprintf(shape, sizeof(shape), "Size=%d,wg=%d/%dx%d", Size, cfg::kFan1Block,
           cfg::kFan2Bx, cfg::kFan2By);
  // `seed` in the schema is 0 here: create_matrix takes no randomness.
  emit_json("gaussian_coal", shape, 0, (Size - 1) * 2, kernel_ms, &clk, mean, stdv);
  fprintf(stderr, "gaussian_coal: %.3f ms kernel (%d t-iters), "
          "SM %u-%u MHz of %u\n",
          kernel_ms, Size - 1, clk.min_mhz, clk.max_mhz, clk.boost_mhz);

  timers_free(&t);
  cudaFree(dA);
  cudaFree(dM);
  free(hA);
  free(hM);
  free(hOut);
  return 0;
}
