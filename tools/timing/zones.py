#!/usr/bin/env python3
"""Report device KERNEL-zone spans from a profiler CSV or a new ttrt run.

Spans cover all cores and RISCs. Automatic workload selection is a heuristic
for this artifact; inspect the table or use --kernel for other programs.
See README.md for profiler setup and per-iteration normalization.
"""
import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

WORKLOAD_IDX = {3: 1, 4: 2, 5: 2, 7: 4, 9: 6}


def cluster(pairs):
    """Group overlapping zones; valid only for non-overlapping dispatches."""
    groups = []
    for start, end, risc in sorted(pairs):
        if groups and start <= groups[-1]["end"]:
            g = groups[-1]
            g["end"] = max(g["end"], end)
            g["riscs"].setdefault(risc, []).append(end - start)
        else:
            groups.append({"start": start, "end": end,
                           "riscs": {risc: [end - start]}})
    for i, g in enumerate(groups):
        g["id"] = i
        g["span"] = g["end"] - g["start"]
    return groups



def parse(csv_path, device=None):
    """Return (frequency in MHz, dispatches in cycles) for one device.

    Missing frequency and incomplete zone pairs are errors, since either can
    silently produce a plausible but incorrect duration.
    """
    with open(csv_path) as f:
        header = f.readline()
        m = re.search(r"CHIP_FREQ\[MHz\]:\s*([^,]+)", header)
        if not m:
            raise ValueError("CSV has no CHIP_FREQ[MHz]")
        freq = float(m[1])
        if not math.isfinite(freq) or freq <= 0:
            raise ValueError("CSV frequency must be finite and positive")
        rdr = csv.reader(f)
        next(rdr, None)
        stacks = defaultdict(list)
        pairs = []
        devices = set()
        for line, r in enumerate(rdr, 3):
            if len(r) < 12:
                if r and any(v.strip() for v in r):
                    raise ValueError(f"malformed CSV row {line}")
                continue
            chip, risc, zone, typ = (r[i].strip() for i in (0, 3, 10, 11))
            if zone not in (f"{risc}-KERNEL", f"{risc.split('_')[0]}-KERNEL"):
                continue
            if device is not None and chip != str(device):
                continue
            devices.add(chip)
            if len(devices) > 1:
                raise ValueError("CSV contains multiple devices; select one with --device")
            if typ not in ("ZONE_START", "ZONE_END"):
                continue
            cyc = int(r[5])
            key = (chip, r[1].strip(), r[2].strip(), risc, zone)
            if typ == "ZONE_START":
                if stacks[key]:
                    raise ValueError(f"unclosed KERNEL zone before row {line}")
                stacks[key].append(cyc)
            else:
                if not stacks[key]:
                    raise ValueError(f"unmatched KERNEL end at row {line}")
                start = stacks[key].pop()
                if cyc < start:
                    raise ValueError(f"KERNEL ends before it starts at row {line}")
                pairs.append((start, cyc, risc))
        if any(stacks.values()):
            raise ValueError("CSV has incomplete KERNEL zones")
    return freq, cluster(pairs)


def select(disp, freq):
    """Cross-check the artifact's dispatch position against TRISC occupancy."""
    live = [d for d in disp if d["span"] / freq >= 1.0]
    k = WORKLOAD_IDX.get(len(live))
    if k is None:
        raise ValueError(f"unrecognized dispatch count {len(live)}; use --kernel")

    def occupancy(d):
        spans = [statistics.median(v) for r, v in d["riscs"].items()
                 if r.startswith("TRISC") and v]
        return max(spans) / d["span"] if spans and d["span"] else 0.0

    ranked = sorted(live, key=occupancy, reverse=True)
    if (occupancy(ranked[0]) == 0 or
            math.isclose(occupancy(ranked[0]), occupancy(ranked[1]), rel_tol=1e-9)):
        raise ValueError("TRISC occupancy is ambiguous; use --kernel")
    if ranked[0]["id"] != live[k]["id"]:
        raise ValueError(f"dispatch count selects {live[k]['id']}, but TRISC occupancy "
                         f"selects {ranked[0]['id']}; use --kernel")
    return live[k]["id"]


def report(disp, freq, kernel, select_workload, iterations=1):
    print(f"\ndevice KERNEL zones: {freq:g} MHz, {len(disp)} dispatch(es)")
    print("   #       span_us     gap_before_us   per-RISC median_us")
    prev_end = None
    for d in disp:
        gap = "" if prev_end is None else f"{(d['start'] - prev_end) / freq:13.2f}"
        per = "  ".join(f"{r}={statistics.median(v) / freq:.2f}"
                        for r, v in sorted(d["riscs"].items()) if v)
        print(f"  {d['id']:2d}  {d['span'] / freq:12.2f} {gap:>13}   {per}")
        prev_end = d["end"]
    if kernel is not None:
        if not re.fullmatch(r"[0-9]+(?::[0-9]+)?", kernel):
            raise ValueError("--kernel must be an index or inclusive range A:B")
        a, _, b = kernel.partition(":")
        lo, hi = int(a), int(b or a)
        if not 0 <= lo <= hi < len(disp):
            raise ValueError(f"--kernel {kernel} is outside 0..{len(disp) - 1}")
    elif select_workload:
        lo = hi = select(disp, freq)
    else:
        return None
    total = sum(d["span"] for d in disp[lo:hi + 1]) / freq
    print(f"\n  dispatches {lo}..{hi}: {total:,.2f} us (sum of spans; excludes gaps)")
    if iterations != 1:
        print(f"  per iteration ({iterations}): {total / iterations:,.2f} us")
    return total / iterations


def capture(args):
    # Resolve runtime and runner from the same active Python installation.
    import ttrt
    runtime = Path(ttrt.__file__).resolve().parent / "runtime"
    if not (runtime / "tracy-capture").is_file():
        raise ValueError(f"{runtime} is not a profiler build; see README.md")
    ttm = Path(args.ttm).resolve()
    if not ttm.is_file():
        raise ValueError(f"no flatbuffer at {ttm}")
    source = runtime / "generated/profiler/.logs/profile_log_device.csv"
    logdir = Path(tempfile.mkdtemp(prefix="zones_"))
    print(f"  capture directory: {logdir}", flush=True)
    # Archive the previous CSV so a failed or empty capture cannot reuse it.
    if source.exists():
        shutil.move(str(source), str(logdir / "previous.csv"))
    if args.reset_device is not None:
        smi = shutil.which("tt-smi")
        if smi is None:
            raise ValueError("--reset-device requires tt-smi on PATH")
        subprocess.run([smi, "-r", str(args.reset_device)], check=True)
    env = dict(os.environ, TT_METAL_DEVICE_PROFILER="1", TT_METAL_LOGGER_LEVEL="WARN")
    if args.init == "diag-dominant":
        runner = Path(__file__).resolve().parents[1] / "run/run-gaussian.py"
        cmd = [sys.executable, str(runner), str(ttm), "--seed", str(args.seed),
               "--out", str(logdir / "artifacts")]
    else:
        cmd = [sys.executable, "-m", "ttrt", "run", str(ttm), "--init", args.init,
               "--seed", str(args.seed)]
    with (logdir / "run.log").open("w") as log:
        result = subprocess.run(cmd, env=env, cwd=logdir, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError(f"ttrt run failed (rc={result.returncode}); see {logdir}/run.log")
    if not source.is_file():
        raise ValueError(f"no new profiler CSV; see {logdir}/run.log")
    saved = logdir / "profile_log_device.csv"
    shutil.copy2(source, saved)
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ttm", nargs="?", help="flatbuffer; omit with --csv")
    ap.add_argument("--csv", type=Path, help="parse a saved CSV without running the device")
    ap.add_argument("--init", default="randn", help="ttrt initializer, or diag-dominant for Gaussian")
    ap.add_argument("--seed", type=int, default=42)
    choice = ap.add_mutually_exclusive_group()
    choice.add_argument("--kernel", help="dispatch index or inclusive range A:B")
    choice.add_argument("--no-select", action="store_true", help="print dispatch table only")
    ap.add_argument("--iterations", type=int, default=1, help="divide selected time by this count")
    ap.add_argument("--device", type=int, help="PCIe slot in CSV; does not select the runtime device")
    ap.add_argument("--reset-device", type=int, help="explicitly reset this device before capture")
    ap.add_argument("--no-reset", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if bool(args.csv) == bool(args.ttm):
        ap.error("give a flatbuffer or --csv, not both")
    if args.iterations < 1:
        ap.error("--iterations must be positive")
    if args.reset_device is not None and (args.csv or args.no_reset or args.reset_device < 0):
        ap.error("--reset-device requires a live capture and cannot be combined with --no-reset")
    freq, disp = parse(args.csv or capture(args), args.device)
    if not disp:
        raise ValueError("no KERNEL zones in the CSV")
    report(disp, freq, args.kernel, not args.no_select, args.iterations)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
