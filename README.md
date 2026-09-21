# Annotation-Driven Migration of CUDA Programs to Tenstorrent Blackhole

The paper presents an MLIR-based compiler pipeline for a statically shaped affine subset of CUDA. Choices that the source does not determine, such as processor assignment, guard realization, and buffer streaming, are expressed as declarative annotations. The compiler checks these annotations and lowers them through the intermediate representations included in this artifact, so each reported number can be traced to the program that produced it.

Workloads are Gaussian elimination, a five-point Jacobi stencil, and a symmetric rank-`k` update, on a 10x10 core grid: `gaussian` at N=5120 (5119 pivots), `jacobi_2d` at N=3200 (1000 time steps), and `syrk` at N=3200.

This repository is a snapshot of those lowerings: for every configuration the paper reports, the intermediate representations at each stage of the pipeline and the compiled program the measurement ran on, with the scripts that reproduce its time and its accuracy.

## Contents

Three groups, one directory per measured configuration:

| directory | what it holds |
|---|---|
| `incremental-realization-ladders/` | the cumulative policy paths of Tables II--IV, one directory per rung and precision |
| `realization-interactions/` | the pairs of Figure 4, grouped by the context that reverses which realization is faster |
| `a100-device-resident-comparison/` | the Blackhole side of Table V, with DRAM--L1 transfers folded into the workload generic |

Also `cuda-sources/` (the CUDA the pipeline starts from), `shared-inputs/` (the semantic front each ladder is authored on), `a100-hand-optimized/` (the CUDA the A100 column was measured on), and `system_desc.ttsys` (the board the programs were compiled against).

Every script is under `tools/`:

    tools/lower/      // rebuild a program from its 70-final-d2m.mlir
    tools/run/        // run the one workload ttrt cannot initialize
    tools/timing/     // reproduce a time
    tools/accuracy/   // reproduce an error

Checkpoint numbers are the same in every directory:

| file | stage |
|---|---|
| `15-scheduled-tileplan.mlir` | scheduled TilePlan, where a loop schedule is the policy |
| `20-policy-summary.md` | which annotations the configuration asks for |
| `21-policy-annotations.diff` | those annotations, as a diff against the shared front |
| `22-annotated-tileflow.mlir` | the authored policy, as the compiler received it |
| `23-realized-pins.diff` | attributes stamped after realization, where a configuration has them |
| `30-realized-tileflow.mlir` | policy-realized TileFlow |
| `40-compute-dm-tasks.mlir` | compute/DM task formation |
| `50-synchronized-tasks.mlir`, `51-buffer-depth.mlir` | synchronization and buffering depth |
| `60-resource-bindings.mlir` | CB and L1 resource binding |
| `70-final-d2m.mlir` | the final D2M program |
| `80-program.ttm` | the compiled TTMetal program the measurements were taken on |
| `REPRO.sh` | the commands that lowered this directory, from the authored annotations down; needs `to-tt` and `task-opt`, which this artifact does not ship |

## Running and rebuilding

`80-program.ttm` needs no custom dialect: it is a TTMetal flatbuffer, and `ttrt` runs it.

    ttrt run --init randn --seed 42 <dir>/80-program.ttm            # jacobi_2d, syrk
    tools/run/run-gaussian.py <dir>/80-program.ttm [--seed 42]      # gaussian

Note that `gaussian` eliminates without pivoting, so it needs a diagonally dominant input that `ttrt`'s initializers cannot express. `run-gaussian.py` supplies the one used for the measurements.

To rebuild `80-program.ttm` from `70-final-d2m.mlir`, build the [casys-utokyo/tt-mlir](https://github.com/casys-utokyo/tt-mlir) fork this work is based on, using branch `waccpd26`, and point `TTMLIR_BIN` at its `build/bin`:

    tools/lower/lower.sh                <dir>/70-final-d2m.mlir <outdir>
    FIDELITY=HiFi2 tools/lower/lower.sh <syrk dir>/70-final-d2m.mlir <outdir>

Each writes `out.ttm` against the `system_desc.ttsys` in this directory; set
`SYSDESC` for another.

`REPRO.sh` in each directory is there to be read: it is how that configuration
was lowered, written out as the commands that ran -- the authored annotations
replayed onto a shared front, then each compiler stage, in order. It needs
`to-tt` to run, which is not part of this snapshot. `gaussian` alternates
between annotating and lowering, because some of the ops it annotates are
created by an earlier stage; the script spells that out step by step.

To reproduce a measurement:

    tools/timing/zones.py     <program.ttm>    # the workload dispatch's device zone
    tools/accuracy/check.py   <dir>            # error against a float64 reference
    tools/accuracy/emulate.py <dir>            # BF16 output, compared bit for bit

Each has a README with what it measures and what it needs.

## License

MIT; see `LICENSE`. `cuda-sources/` and `a100-hand-optimized/` contain code from
PolyBench/GPU (`jacobi_2d`, `syrk`) and Rodinia (`gaussian`), which remains under
its own terms; each directory's README says what came from where.

## Publication

    Ayumi Ohno and Shinya Takamaeda-Yamazaki, "Annotation-Driven Migration of CUDA Programs to Tenstorrent Blackhole,"
    Workshop on Accelerator Programming and Directives (WACCPD 2026),
    held in conjunction with SC (Accepted).
