# Accuracy checks

Run with the Python environment containing `ttrt`, `torch`, and `numpy`. Each
directory must hold `70-final-d2m.mlir` and `80-program.ttm`; the workload and
the depth are read from the IR. Both scripts exit nonzero on any failure.

    python tools/accuracy/check.py   <dir>        # error against a float64 reference
    python tools/accuracy/emulate.py <bf16-dir>   # BF16 output, bit for bit

## `check.py`

`relFro` is `||output - reference||_F / ||reference||_F` over the region the
algorithm writes:

| Workload | Region | Also reported |
|---|---|---|
| Jacobi | interior `[1:-1, 1:-1]` | the border, which must still equal the input |
| Gaussian | off-diagonal | the diagonal, whose magnitude would dominate a whole-matrix norm |
| SYRK | whole output | — |

The reference runs in FP64 and is never narrowed. Jacobi and SYRK start from
the inputs `ttrt` produced, which exist only as the device received them.
Gaussian's input is generated here from `--seed`, so its reference starts from
the FP32 matrix that was specified, not from the BF16 copy the device was
handed: representing that matrix is part of the error a BF16 program makes, and
the reported figure includes it. Gaussian uses a diagonally dominant matrix
because it does not pivot, and the run rejects a capture that was not given that
matrix. Cost: seconds for SYRK, two minutes for Jacobi, half a minute for
Gaussian's 5119 eliminations.

The paper's errors belong to the fastest configuration at each precision; other
rungs of the same ladder score differently.

⚠ The 100-iteration SYRK binary under `a100-device-resident-comparison/`
overflows BF16 by construction -- it is a timing experiment, and `check.py`
rejects it.

## `emulate.py`

`tensix.py` is the Blackhole compute units as measured. `emulate.py` runs the
workload through them and compares BF16 encodings over the whole output,
including signed zeros: bit-exact, or a count of the encodings that differ.

It models the configuration whose error the paper reports for each workload;
other rungs emit different arithmetic and will mismatch.

Jacobi needs `--frame`: `sender` (the default) is the add tree with one SFPU
addition, `receiver` four sequential FPU additions.

The model runs on the host. SYRK's output splits into 32-row bands that are
independent, so `--workers` is how many of them are evaluated at once; the other
two workloads are sequential in their depth. `--steps N` shortens the model only
and then compares nothing. Cost: SYRK two minutes over eight workers, Jacobi
twenty, Gaussian thirty.

## Reusing saved tensors

    python tools/accuracy/check.py   <dir> --keep /tmp/capture
    python tools/accuracy/emulate.py <dir> --artifacts /tmp/capture

`--keep` needs an empty directory; `--artifacts` skips the device. ⚠ Use a
capture from the configuration you are naming -- the tensors do not record which
binary produced them.
