# CUDA inputs

The three programs the compiler consumes. `shared-inputs/<workload>/00-cuda-input.mlir`
is cgeist's output for the file named here, and everything else in this artifact is
downstream of that.

| file | workload | upstream |
|---|---|---|
| `gaussian.cu` | Gaussian elimination | [Rodinia](https://github.com/yuhc/gpu-rodinia/blob/master/cuda/gaussian/gaussian.cu) |
| `jacobi_2d.cu`, `jacobi2D.cuh` | five-point Jacobi stencil | [PolyBench-ACC](https://github.com/cavazos-lab/PolyBench-ACC/blob/master/CUDA/stencils/jacobi-2d-imper/jacobi2D.cu) |
| `syrk.cu`, `syrk.cuh` | symmetric rank-`k` update | [PolyBench-ACC](https://github.com/cavazos-lab/PolyBench-ACC/blob/master/CUDA/linear-algebra/kernels/syrk/syrk.cu) |

## Everything that differs from upstream

The three input adaptations are the ones the paper states in Section II:

* `gaussian.cu` omits the update of the right-hand-side vector `b`, which leaves
  `Fan2`'s signature and its one statement. `Fan1` is untouched and the rest of
  `Fan2` is too, including the offset indexing `A[x+t+1, y+t]` and its guards,
  which are upstream's.
* `syrk.cu` replaces PolyBench's in-place reduction with a scalar accumulator and
  a single final update, `C = alpha*acc + beta*C`. Equivalent in exact
  arithmetic; it changes rounding.
* the thread block is `32x32` in all three, because the lowering maps the threads
  of one block onto the elements of one Tensix tile. Upstream's PolyBench headers
  say `32x8`.

Everything else is mechanical. The shapes are compile-time constants -- `N` is
`10*32*10` in both headers with `TSTEPS` 1000, and `gaussian.cu` fixes `Size` at
`16*32*10` where upstream reads a matrix from a file -- `alpha` and `beta` are
`2.0f` and `3.0f`, and the drivers are gone: initialization, the CPU reference,
timing, printing, argument parsing. What remains of each host function is a
launch site, taking the device pointers as parameters.

`jacobi_2d.cu`'s kernels are upstream's, unchanged.

## How the MLIR was produced

    cgeist <file>.cu --cuda-gpu-arch=sm_75 --resource-dir=<clang resource dir> -S \
        -o <file>.cu.mlir

`shared-inputs/<workload>/00-cuda-input.mlir` is that output with a provenance
header prepended and nothing else changed.
