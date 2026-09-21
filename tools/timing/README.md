# Device timing

Run with the Python environment holding a profiler-enabled `ttrt`:

    python tools/timing/zones.py <program.ttm>

It prints every dispatch span and then the workload's. `--help` lists the rest.

## Getting a program the profiler will accept

Build ttrt with `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`, then re-lower against its
descriptor:

    ttrt query --save-artifacts
    SYSDESC=<query-output>/system_desc.ttsys tools/lower/lower.sh <dir>/70-final-d2m.mlir /tmp/perf
    python tools/timing/zones.py /tmp/perf/out.ttm

`FIDELITY=HiFi2` when lowering SYRK, `--init diag-dominant` when timing Gaussian.

## What the number is

A dispatch span is `max(KERNEL end) - min(KERNEL start)` across cores and RISCs,
converted from cycles with the CSV's `CHIP_FREQ[MHz]`. Host submission, transfers
outside those zones, and the gaps between dispatches are excluded. This is
tt-metal's own `device_kernel_duration` aggregation.

## Which dispatch

Layout conversion and DRAM transfers are dispatches too, so only one of the
several a program enqueues is the workload. Dispatches under 1 us are ignored,
and the workload's position among the rest is fixed by how many there are: 3 put
it at position 1, 4 at 2, 5 at 2, 7 at 4, 9 at 6. That position is then checked
against which dispatch has the busiest TRISCs; a disagreement is an error.

⚠ Position, not index. Some runs enqueue an empty generic ahead of the workload
-- 0.03 us, one RISC reporting -- so the same build can put the workload at index
2 in four runs and at index 3 in the fifth.

`--kernel 4` or `--kernel 2:4` selects by hand; `--no-select` prints the table
and stops. Dispatches are grouped by zone overlap, which assumes they run
serially.

## Reporting it

Output is in microseconds and the paper's tables are in milliseconds. The paper
reports five-run means: capture five runs and average them. For the
device-resident comparison's SYRK, whose dispatch runs the body 100 times, divide
by that count with `--iterations 100`; everything else is `--iterations 1`.
