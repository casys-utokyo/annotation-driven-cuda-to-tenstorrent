#!/usr/bin/env python3
"""Report error against an FP64 reference using the device's saved inputs.

Run with the Python environment containing ttrt and torch. See README.md for
supported configurations, comparison regions, and expected errors.
"""
import argparse
import math
import os
import re
import sys
import tempfile
from pathlib import Path

def read_ir(d):
    return (d / "70-final-d2m.mlir").read_text()


def workload_of(d):
    ir = read_ir(d)
    for symbol, workload in (("runJacobi", "jacobi"), ("ForwardSub", "gaussian"),
                             ("syrk_kernel", "syrk")):
        if re.search(r"func\.func @\S*" + symbol, ir):
            return workload
    raise ValueError(f"cannot identify workload in {d}/70-final-d2m.mlir")


def steps_of(d):
    """Read the shared temporal depth of the emitted synchronized loops."""
    found = {int(m) for m in re.findall(r"d2m\.sync_loop = (\d+) : i64", read_ir(d))}
    if len(found) != 1 or min(found) < 1:
        raise ValueError(f"{d}: expected one positive d2m.sync_loop depth, found {sorted(found)}")
    return found.pop()


def validate_config(d, workload):
    ir = read_ir(d)
    # The comparison binary feeds C back 100 times and can overflow BF16.
    # Its output is not the single-call accuracy experiment.
    if workload == "syrk" and re.search(r"scf\.for %rep\b", ir):
        raise ValueError("accuracy requires a single-call SYRK configuration; "
                         "use incremental-realization-ladders/syrk/04-depth-two-delivery")


def diag_dominant(n, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    a = -0.5 + torch.rand(n, n, generator=g, dtype=torch.float32)
    a[torch.arange(n), torch.arange(n)] += float(n)
    return a


def load_tensors(out):
    """Load one program's tensors; reject ambiguous or incomplete captures."""
    import torch
    outputs = list(out.rglob("device_output_0.pt"))
    if len(outputs) != 1:
        raise ValueError(f"expected one device_output_0.pt under {out}, found {len(outputs)}")
    folder = outputs[0].parent
    ins = sorted(folder.glob("input_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not ins or len(list(folder.glob("device_output_*.pt"))) != 1:
        raise ValueError(f"expected inputs and one output under {folder}")

    def load(p):
        t = torch.load(p, map_location="cpu", weights_only=True)
        # ttrt stores a singleton list for a one-device tensor.
        if isinstance(t, (list, tuple)) and len(t) == 1:
            t = t[0]
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"expected a tensor in {p}")
        return t

    return [load(p) for p in ins], load(outputs[0])


def run(ttm, out, workload, seed):
    """Run program 0 and load its saved tensors without changing dtype."""
    from ttrt.common.api import API
    from ttrt.common.run import Run
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"artifact directory must be empty: {out}")
    API.initialize_apis()
    init = "randn"
    if workload == "gaussian":
        def custom(self, shape, dtype):
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(f"expected a square Gaussian input, got {shape}")
            return diag_dominant(shape[0], seed).to(dtype)

        Run.TorchInitializer.custom = custom
        if "custom" not in Run.TorchInitializer.init_fns:
            Run.TorchInitializer.init_fns = sorted(Run.TorchInitializer.init_fns + ["custom"])
        init = "custom"
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "WARN")
    rc, _ = Run(args={"binary": str(ttm), "--init": init, "--seed": seed,
                      "--program-index": 0, "--save-artifacts": True,
                      "--artifact-dir": str(out)})()
    if rc != 0:
        raise RuntimeError(f"ttrt run failed (rc={rc}); artifacts: {out}")
    return load_tensors(out)


def start(workload, ins, seed):
    """The matrix the algorithm was asked to transform.

    Gaussian's input is generated here, so the reference starts from the FP32
    matrix that was specified rather than from the copy the device received:
    representing that matrix is part of the error a BF16 program makes, and the
    reported figure includes it. The other workloads' inputs are produced by
    ttrt and exist only as the device got them.
    """
    import torch
    if workload != "gaussian":
        return ins[0]
    a = diag_dominant(ins[0].shape[0], seed)
    if not torch.equal(a.to(ins[0].dtype), ins[0]):
        raise ValueError("the device input is not diag_dominant(n, seed): wrong "
                         "--seed, or a capture from a different configuration")
    return a


def reference(workload, ins, steps, seed):
    """Recompute in FP64, never narrowing, from the inputs that were specified."""
    import torch
    expected = 2 if workload == "syrk" else 1
    if len(ins) != expected:
        raise ValueError(f"{workload} needs {expected} inputs, got {len(ins)}")
    a = start(workload, ins, seed).to(torch.float64).clone()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square matrix, got {tuple(a.shape)}")
    if workload == "jacobi":
        for _ in range(steps):
            b = a.clone()
            b[1:-1, 1:-1] = 0.2 * (a[1:-1, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] +
                                   a[2:, 1:-1] + a[:-2, 1:-1])
            a = b
        return a
    if workload == "syrk":
        c = ins[1].to(torch.float64)
        if c.shape != a.shape:
            raise ValueError("SYRK A and C shapes differ")
        return 2.0 * (a @ a.T) + 3.0 * c
    if workload != "gaussian" or not 0 < steps < a.shape[0]:
        raise ValueError(f"unsupported workload/depth: {workload}/{steps}")
    for t in range(steps):
        piv = a[t, t]
        if not torch.isfinite(piv) or piv == 0:
            raise ValueError(f"invalid Gaussian pivot at step {t}")
        m = (a[t + 1:, t] / piv).unsqueeze(1)
        # Fan2 updates only columns at or to the right of the pivot.
        a[t + 1:, t:] -= m * a[t, t:].unsqueeze(0)
    return a


def region(workload, t):
    import torch
    if workload == "jacobi":
        m = torch.zeros_like(t, dtype=torch.bool)
        m[1:-1, 1:-1] = True
        return m, "interior", "border"
    if workload == "gaussian":
        return ~torch.eye(t.shape[0], dtype=torch.bool), "off-diag", "diagonal"
    return torch.ones_like(t, dtype=torch.bool), "whole", None


def ratio(numerator, denominator):
    if denominator == 0:
        return 0.0 if numerator == 0 else math.inf
    return numerator / denominator


def relfro(got, ref, mask):
    return ratio(float((got - ref)[mask].norm()), float(ref[mask].norm()))


def measure(workload, ins, got, steps, seed):
    import torch
    if not all(torch.isfinite(t).all() for t in [*ins, got]):
        raise ValueError("non-finite device input or output")
    dtype = str(got.dtype).split(".")[-1]
    ref = reference(workload, ins, steps, seed)
    got = got.to(torch.float64)
    if got.shape != ref.shape:
        raise ValueError(f"output shape {tuple(got.shape)} != reference {tuple(ref.shape)}")
    if not torch.isfinite(ref).all():
        raise ValueError("non-finite FP64 reference")
    mask, name, rest = region(workload, ref)
    if not mask.any():
        raise ValueError("empty comparison region")
    error = relfro(got, ref, mask)
    print(f"  workload={workload} dtype={dtype} steps={steps}")
    print(f"  relFro ({name}) = {error:.6e}")
    if rest:
        print(f"  relFro ({rest}) = {relfro(got, ref, ~mask):.6e}")
    if workload == "jacobi" and not torch.equal(got[~mask], ref[~mask]):
        raise ValueError("Jacobi border differs from the input")
    return error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep", type=Path, help="save new artifacts in an empty directory")
    ap.add_argument("--artifacts", type=Path, help="use saved tensors instead of running the device")
    a = ap.parse_args()
    if a.keep and a.artifacts:
        ap.error("--keep and --artifacts are mutually exclusive")
    d = a.dir.resolve()
    w = workload_of(d)
    validate_config(d, w)
    steps = steps_of(d) if w != "syrk" else 1
    if a.artifacts:
        ins, got = load_tensors(a.artifacts)
    else:
        ttm = d / "80-program.ttm"
        if not ttm.is_file():
            raise ValueError(f"no flatbuffer at {ttm}")
        with tempfile.TemporaryDirectory(prefix="accuracy_") as tmp:
            ins, got = run(ttm, a.keep or Path(tmp), w, a.seed)
    measure(w, ins, got, steps, a.seed)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(str(exc))
