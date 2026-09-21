#!/usr/bin/env python3
"""Run a `gaussian` program on a diagonally dominant input.

    ./run/run-gaussian.py <dir>/80-program.ttm [--seed 42] [--out DIR]

`ttrt`'s initializers cannot express one, so the tensor goes in through the
Python API instead.
"""
import argparse
import os
import sys
import tempfile

try:
    import torch
    from ttrt.common.api import API
    from ttrt.common.run import Run
except ImportError as exc:  # not an installation problem -- the wrong interpreter
    sys.exit(f"{exc}\n"
             "Run this with the Python that `ttrt` is installed in, e.g.\n"
             "    <ttrt venv>/bin/python run/run-gaussian.py <dir>/80-program.ttm")


def diag_dominant(n, seed):
    g = torch.Generator().manual_seed(seed)
    a = -0.5 + torch.rand(n, n, generator=g, dtype=torch.float32)
    idx = torch.arange(n)
    a[idx, idx] += float(n)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ttm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=5120, help="matrix order")
    ap.add_argument("--out", default=None, help="artifact dir (default: a temp dir)")
    a = ap.parse_args()

    API.initialize_apis()

    tensor = diag_dominant(a.n, a.seed)

    # ttrt picks its input by name, so the tensor goes in as a new initializer.
    def _custom(self, shape, dtype):
        t = tensor
        if tuple(t.shape) != tuple(shape):
            t = t.reshape(shape)
        return t.to(dtype)

    Run.TorchInitializer.custom = _custom
    if "custom" not in Run.TorchInitializer.init_fns:
        Run.TorchInitializer.init_fns = sorted(
            Run.TorchInitializer.init_fns + ["custom"])

    out = a.out or tempfile.mkdtemp(prefix="gaussian_run_")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "WARN")
    rc, _ = Run(args={"binary": a.ttm, "--init": "custom",
                      "--save-artifacts": True, "--artifact-dir": out})()
    print(f"artifacts in {out}", file=sys.stderr)
    raise SystemExit(rc)


main()
