#!/usr/bin/env python3
"""Compare a BF16 device output with the Tensix arithmetic model.

A full-depth mismatch exits nonzero. --steps with a different depth exercises
only the model and does not compare outputs. See README.md for model scope.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

from check import load_tensors, run, steps_of, validate_config, workload_of


def compare(emu, dev):
    """Compare BF16 encodings, including the sign of zero."""
    import numpy as np
    import torch
    if tuple(emu.shape) != tuple(dev.shape):
        raise ValueError(f"model shape {emu.shape} != device shape {tuple(dev.shape)}")
    if not np.isfinite(emu).all() or not torch.isfinite(dev).all():
        raise ValueError("non-finite model or device output")
    ev = torch.from_numpy(np.asarray(emu, dtype=np.float32))
    bf = ev.to(torch.bfloat16)
    if not torch.equal(ev, bf.float()):
        raise ValueError("model produced values not representable in BF16")
    same = bf.contiguous().view(torch.int16) == dev.contiguous().view(torch.int16)
    eq, n = int(same.sum()), same.numel()
    print(f"  bitwise match {eq:,}/{n:,} = {100 * eq / n:.6f}%")
    print(f"  max|diff| {float((dev.double() - ev.double()).abs().max()):.3e}")
    if eq != n:
        raise ValueError(f"NOT BIT-EXACT: {n - eq:,} BF16 encodings differ")
    print("  BIT-EXACT")


def jacobi_model(a0, steps, frame, **_):
    import tensix as tx
    return tx.jacobi(a0.float().numpy(), steps, frame=frame)


def gaussian_model(a0, steps, **_):
    """Forward elimination, one measured step at a time.

    The two guards are the emitter's affine masks: rows i >= t+1, columns
    j >= t. The reciprocal is not a compute-unit op -- the mover truncates a
    float32 divide to BF16 -- and the rank-1 update is two broadcast multiplies
    and an FPU subtract.
    """
    import numpy as np
    import tensix as tx
    a = a0.float().numpy().astype(np.float32).copy()
    for t in range(steps):
        piv = float(a[t, t])
        if not np.isfinite(piv) or piv == 0.0:
            raise ValueError(f"invalid pivot {piv} at step {t}")
        a = tx.gaussian_step(a, t)
    return a


def _syrk_band(args):
    import numpy as np
    import tensix as tx
    br, a, c, nt, block = args
    out = np.empty((32, nt * 32), dtype=np.float32)
    for bc in range(nt):
        acc = None
        for start in range(0, nt, block or nt):
            d = None
            for bk in range(start, min(start + (block or nt), nt)):
                d = tx.matmul_block_tile(a[32 * br:32 * br + 32, 32 * bk:32 * bk + 32],
                                         a[32 * bc:32 * bc + 32, 32 * bk:32 * bk + 32].T, d)
            acc = d if acc is None else tx.rnd(acc + d, mode="rne")
        out[:, 32 * bc:32 * bc + 32] = tx.syrk_epilogue(
            c[32 * br:32 * br + 32, 32 * bc:32 * bc + 32], acc)
    return br, out


def syrk_model(a0, steps, workers=8, c0=None, block=10, **_):
    """Each output tile walks the k-tiles in emitted order, then the epilogue.

    `block` is where the k reduction accumulates: the packer adds each k-block's
    partial into L1 (the compiler's default), so partials are combined outside
    Dst. Passing 0 models the Dst-resident chain instead, which is what the
    ladder's first two rungs do.
    """
    import multiprocessing as mp
    import numpy as np
    a = a0.float().numpy()
    c = c0.float().numpy()
    nt = a.shape[0] // 32
    out = np.empty_like(a)
    work = [(br, a, c, nt, block) for br in range(nt)]
    with mp.Pool(min(workers, nt)) as pool:
        for br, band in pool.imap_unordered(_syrk_band, work):
            out[32 * br:32 * br + 32] = band
    return out


MODELS = {"jacobi": jacobi_model, "gaussian": gaussian_model, "syrk": syrk_model}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--steps", type=int, help="override model depth; other depths are not compared")
    ap.add_argument("--frame", default="sender", choices=("sender", "receiver"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifacts", type=Path, help="use saved tensors instead of running the device")
    ap.add_argument("--workers", type=int, default=8, help="SYRK only: output-tile bands in parallel")
    a = ap.parse_args()
    if a.steps is not None and a.steps < 1:
        ap.error("--steps must be positive")

    import numpy as np
    import torch
    import tensix as tx

    d = a.dir.resolve()
    workload = workload_of(d)
    if workload not in MODELS:
        raise ValueError(f"no model for {workload}")
    validate_config(d, workload)
    depth = steps_of(d) if workload != "syrk" else 1
    steps = depth if a.steps is None else a.steps
    if a.artifacts:
        ins, dev = load_tensors(a.artifacts)
    else:
        ttm = d / "80-program.ttm"
        if not ttm.is_file():
            raise ValueError(f"no flatbuffer at {ttm}")
        with tempfile.TemporaryDirectory(prefix="emulate_") as tmp:
            ins, dev = run(ttm, Path(tmp), workload, a.seed)
    want = 2 if workload == "syrk" else 1
    if len(ins) != want or dev.dtype != torch.bfloat16:
        raise ValueError(f"expected {want} input(s) and a BF16 output")
    if any(t.dtype != torch.bfloat16 for t in ins):
        raise ValueError("this is not a BF16 configuration; the model is BF16")
    a0 = ins[0]
    if a0.ndim != 2 or min(a0.shape) < 3 or dev.shape != a0.shape:
        raise ValueError("expected matching 2D stencil input and output")
    if not torch.isfinite(a0).all() or not torch.isfinite(dev).all():
        raise ValueError("non-finite input or device output")
    print(f"  {workload}: shape={tuple(a0.shape)} device steps={depth} "
          f"model steps={steps}" + (f" frame={a.frame}" if workload == "jacobi" else ""),
          flush=True)
    if steps != depth:
        print("  MODEL ONLY: depth differs; no accuracy comparison will be made", flush=True)
    t0 = time.monotonic()
    emu = MODELS[workload](a0, steps, frame=a.frame, workers=a.workers,
                           c0=ins[1] if workload == "syrk" else None)
    print(f"  tensix model: {time.monotonic() - t0:.0f} s", flush=True)
    if not np.isfinite(emu).all():
        raise ValueError("non-finite model output")
    if steps == depth:
        compare(emu, dev)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(str(exc))
