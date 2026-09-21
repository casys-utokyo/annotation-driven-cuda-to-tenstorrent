#ifndef A100_MEASURE_H
#define A100_MEASURE_H

#include <cuda_runtime.h>
#include <nvml.h>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d: CUDA error: %s (%s)\n", __FILE__, __LINE__,      \
              cudaGetErrorString(_e), #expr);                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Fails loudly on a bad launch config / kernel fault, which otherwise shows up
// only as a suspiciously fast measurement.
static inline void check_launch(const char *what) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "kernel launch failed (%s): %s\n", what,
            cudaGetErrorString(e));
    exit(1);
  }
}

// LCG so two runs with the same seed produce identical inputs. Same generator
// as gpu-rodinia-bench/gaussian_bench.cu.
static inline unsigned next_lcg(unsigned *st) {
  *st = (*st * 1103515245u + 12345u) & 0x7fffffffu;
  return *st;
}

// Uniform in [-0.5, 0.5).
static inline void fill_uniform(float *buf, size_t n, unsigned seed) {
  unsigned s = seed ? seed : 1;
  for (size_t i = 0; i < n; i++)
    buf[i] = (next_lcg(&s) % 10000) / 10000.0f - 0.5f;
}

// Uniform off-diagonal + boosted diagonal. Gaussian elimination divides by
// a[t,t], so the pivots must never reach zero.
static inline void fill_diag_dominant(float *buf, int n, unsigned seed) {
  unsigned s = seed ? seed : 1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      buf[(size_t)i * n + j] = (next_lcg(&s) % 10000) / 10000.0f - 0.5f;
    buf[(size_t)i * n + i] += (float)n;
  }
}

// Checksum of the result, for eyeballing against the TT-side output.
static inline void summarize(const float *buf, size_t n, double *mean,
                             double *stdv) {
  double sum = 0, sumsq = 0;
  for (size_t i = 0; i < n; i++) {
    sum += buf[i];
    sumsq += (double)buf[i] * buf[i];
  }
  *mean = sum / n;
  double var = sumsq / n - (*mean) * (*mean);
  *stdv = sqrt(var > 0 ? var : 0);
}

// ---------------------------------------------------------------------------
// Warm-up: a CLOCK RAMP, measured in milliseconds, not iterations.
//
// This card idles its SMs well below boost and needs sustained load to reach
// it. That is the whole job of the warm-up here -- not caches, not TLBs. Keep
// the budget in TIME so one value covers all three workloads; an iteration
// count would need a different number per workload, because one rep is
// sub-millisecond on syrk and over a second on gaussian.
inline double g_warmup_ms = 1000.0;  // --warmup-ms overrides

template <typename F>
static inline void warmup(F step, double budget_ms = -1.0) {
  if (budget_ms < 0) budget_ms = g_warmup_ms;
  auto t0 = std::chrono::steady_clock::now();
  do {
    step();
    cudaDeviceSynchronize();
  } while (std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
               .count() < budget_ms);
}

// Returns 1 if it consumed argv[*i] (and possibly argv[*i+1]). Call it first in
// each driver's flag loop so every driver spells the knob the same way.
static inline int parse_warmup_flag(int argc, char **argv, int *i) {
  if (!strcmp(argv[*i], "--warmup-ms") && *i + 1 < argc) {
    g_warmup_ms = atof(argv[++(*i)]);
    if (g_warmup_ms < 0) g_warmup_ms = 0;
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Clock guard.
//
// Sample the SM clock during the measurement and report it: on this card a
// changed time has always been the clock moving rather than the kernel.
// The NVML handle comes from the PCI bus id rather than the device index, so it
// still points at the right GPU under CUDA_VISIBLE_DEVICES.
// ---------------------------------------------------------------------------
struct ClockWatch {
  nvmlDevice_t dev;
  unsigned int boost_mhz, min_mhz, max_mhz;
  double sum_mhz;
  int n, ok;
};

static inline void clockwatch_init(ClockWatch *c) {
  memset(c, 0, sizeof(*c));
  c->min_mhz = 1u << 30;
  if (nvmlInit_v2() != NVML_SUCCESS) return;
  int d = 0;
  if (cudaGetDevice(&d) != cudaSuccess) return;
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, d) != cudaSuccess) return;
  char pci[32];
  snprintf(pci, sizeof(pci), "%08X:%02X:%02X.0", p.pciDomainID, p.pciBusID,
           p.pciDeviceID);
  if (nvmlDeviceGetHandleByPciBusId_v2(pci, &c->dev) != NVML_SUCCESS) return;
  if (nvmlDeviceGetMaxClockInfo(c->dev, NVML_CLOCK_SM, &c->boost_mhz) !=
      NVML_SUCCESS)
    return;
  c->ok = 1;
}

// Call once per rep, OUTSIDE the timed region.
static inline void clockwatch_sample(ClockWatch *c) {
  if (!c->ok) return;
  unsigned int mhz = 0;
  nvmlDeviceGetClockInfo(c->dev, NVML_CLOCK_SM, &mhz);
  c->sum_mhz += mhz;
  c->n++;
  if (mhz < c->min_mhz) c->min_mhz = mhz;
  if (mhz > c->max_mhz) c->max_mhz = mhz;
}

struct Timers {
  cudaEvent_t a, b;
};

static inline void timers_init(Timers *t) {
  CUDA_CHECK(cudaEventCreate(&t->a));
  CUDA_CHECK(cudaEventCreate(&t->b));
}

static inline void timers_free(Timers *t) {
  cudaEventDestroy(t->a);
  cudaEventDestroy(t->b);
}

static inline void timer_start(Timers *t) { CUDA_CHECK(cudaEventRecord(t->a)); }

static inline float timer_stop_ms(Timers *t) {
  CUDA_CHECK(cudaEventRecord(t->b));
  CUDA_CHECK(cudaEventSynchronize(t->b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t->a, t->b));
  return ms;
}

// One JSON object per run on stdout, and ONE RUN PER PROCESS. bench.py collects
// them; everything else the drivers print goes to stderr.
//
// One process emits one measurement: `kernel_ms` is a single timed span, not a
// statistic. Dispersion is bench.py's job, across the processes it runs.
// `clk_mhz_min` against `boost_mhz` says whether the run sat at the operating
// point it thinks it did.
static inline void emit_json(const char *workload, const char *shape, int seed,
                             int launches_per_rep, double kernel_ms,
                             const ClockWatch *clk, double out_mean,
                             double out_std) {
  int lpr = launches_per_rep > 0 ? launches_per_rep : 1;
  double km = kernel_ms;
  printf("{\"workload\":\"%s\",\"shape\":\"%s\",\"seed\":%d,"
         "\"launches_per_rep\":%d,\"warmup_ms\":%.0f,"
         "\"kernel_ms\":%.4f,\"kernel_us_per_launch\":%.4f,"
         "\"clk_mhz_min\":%u,\"clk_mhz_mean\":%.0f,\"boost_mhz\":%u,"
         "\"out_mean\":%.6f,\"out_std\":%.6f}\n",
         workload, shape, seed, lpr, g_warmup_ms, km, km * 1000.0 / lpr,
         clk->ok ? clk->min_mhz : 0u, clk->n ? clk->sum_mhz / clk->n : 0.0,
         clk->boost_mhz, out_mean, out_std);
  fflush(stdout);
}

#endif  // A100_MEASURE_H
