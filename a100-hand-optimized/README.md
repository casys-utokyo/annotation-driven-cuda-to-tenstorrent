# A100 hand-optimized sources

The CUDA the paper's A100 column was measured on. Three programs, one per
workload:

| Program | Workload | What it does |
|---|---|---|
| `jacobi_2d_pingpong.cu` | Jacobi 2D | Ping-pong: swaps the two buffers each timestep instead of running a copy-back kernel. |
| `gaussian_opt.cu` | Gaussian elimination | Coalescing: puts the fast-varying thread index on the column rather than the row. |
| `syrk_cublas.cu` | SYRK | A cuBLAS tensor-core call. Not a hand optimization -- the vendor library is already the tuned path. |

## Build and run

    make
    python3 bench.py

`bench.py` reports kernel time, its run-to-run spread, and the SM clock each run
observed. It starts a fresh process per trial -- five by default, `--trials` to
change it -- in randomised order and pinned to one NUMA node, so the spread it
prints is process-to-process, not kernel-to-kernel. Each program times its own
kernel and reads the clock once the timed region closes; `results.json` collects
both.

The programs take no shape flags: every block shape and schedule constant is a
`cfg::` block in the .cu that uses it, so what ships is what was measured.

## Provenance

`jacobi_2d_pingpong.cu` starts from PolyBench/GPU and `gaussian_opt.cu` from
Rodinia -- the same programs `../cuda-sources/` feeds the compiler, at the same
shapes, which is what makes the two columns comparable; each marks the regions
that are upstream verbatim. `syrk_cublas.cu` shares only the shape and the
driver: its compute is a cuBLAS call, not a kernel of ours or PolyBench's.

## Environment

Measured on an A100-PCIE-40GB (sm_80) at a 250 W cap with CUDA 12.0, built
`-O3 -arch=sm_80`. Kernel time is what is quoted throughout; there is no
end-to-end column.
