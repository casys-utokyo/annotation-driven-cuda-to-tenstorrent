NVCC ?= nvcc
ARCH ?= sm_80
NVCCFLAGS ?= -O3 -arch=$(ARCH) -I.
# measure.h samples the SM clock through NVML, hence -lnvidia-ml.
LDLIBS_COMMON := -lnvidia-ml

BIN := jacobi_2d_pingpong gaussian_opt syrk_cublas

# No build knobs. Every shape and schedule constant is a `cfg::` block in the
# .cu that uses it; each binary takes only --device and --warmup-ms.

all: $(BIN)

jacobi_2d_pingpong: jacobi_2d_pingpong.cu measure.h
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS_COMMON)

gaussian_opt: gaussian_opt.cu measure.h
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS_COMMON)

# The one binary here whose compute path is not a hand optimization: the vendor
# library is already the tuned path. It lives here so all three workloads share
# one driver structure. Needs -lcublas.
syrk_cublas: syrk_cublas.cu measure.h
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas $(LDLIBS_COMMON)

clean:
	rm -f $(BIN)

.PHONY: all run clean
