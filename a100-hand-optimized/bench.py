#!/usr/bin/env python3
"""Run the three hand-optimized rows under a stated, checked protocol.

  python3 bench.py                       # all three -> results.json
  python3 bench.py --only gaussian_coal
  python3 bench.py --trials 8            # tighter estimate
  DEVICE=1 python3 bench.py

"""
import argparse, json, os, random, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

# stat / per_launch: see the module docstring. Each binary does exactly one
# timed span per process -- there is no repeat knob -- and the warm-up budget is
# the driver's (measure.h, 1000 ms), reported back in the JSON.
#
# `per_launch` divides the ms statistics by the run's `launches_per_rep` before
# quoting. Only syrk sets it: its span is `--inner` launches and one run of that
# workload is ONE syrk, whereas one run of jacobi is all 1000 timesteps and one
# run of gaussian is all 5119 pivots. Without this the syrk row would quote the
# 40 ms span.
WORKLOADS = {
    "jacobi_2d_pingpong": dict(
        binary="jacobi_2d_pingpong", args=[],
        stat="kernel_ms",
        note="one stencil launch per timestep; the baseline runs two"),
    "gaussian_coal": dict(
        binary="gaussian_opt", args=[],
        stat="kernel_ms",
        note="Fan2 with threadIdx.x on the column instead of the row"),
    "syrk_cublas": dict(
        binary="syrk_cublas", args=[],
        stat="kernel_ms", per_launch=True,
        note="cuBLAS bf16 GemmEx; 100 back-to-back launches per timed span"),
}

def run_one(name, cfg, device):
    # --warmup-ms is deliberately NOT passed. The warm-up budget belongs to the
    # driver (measure.h's g_warmup_ms, 1000 ms), and passing it from here once
    # produced a recipe that said 400 while two of the three binaries were
    # hard-coded to 1000 and reported 400 in their own JSON.
    cmd = ["numactl", "--cpunodebind=0", "--membind=0",
           os.path.join(HERE, cfg["binary"]),
           "--device", str(device)] + cfg["args"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"{name}: no JSON line\ncmd: {' '.join(cmd)}\n"
                       f"stderr: {p.stderr[-500:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=5,
                    help="fresh processes per workload; each is one measurement "
                         "(default 5)")
    ap.add_argument("--idle", type=float, default=5.0,
                    help="seconds idle before each process, so a measurement "
                         "does not inherit the previous one's thermal state "
                         "(default 5)")
    ap.add_argument("--device", type=int, default=int(os.environ.get("DEVICE", 0)))
    ap.add_argument("--only", action="append", default=None,
                    help="workload name; repeatable")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--seed", type=int, default=0, help="order-shuffle seed")
    args = ap.parse_args()

    names = args.only or list(WORKLOADS)
    for n in names:
        if n not in WORKLOADS:
            sys.exit(f"unknown workload {n}; have {', '.join(WORKLOADS)}")

    subprocess.run(["make", "-s", "-C", HERE, "all"], check=True)
    gpu = subprocess.run(
        ["nvidia-smi", "-i", str(args.device), "--query-gpu=name,compute_cap,"
         "power.limit,clocks.max.graphics", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    print(f"# gpu {args.device}: {gpu}")
    print(f"# {args.trials} fresh processes per workload, one measurement each, "
          f"{args.idle}s idle between, randomised order, numactl-pinned to "
          f"node 0")

    plan = [n for n in names for _ in range(args.trials)]
    random.Random(args.seed).shuffle(plan)

    runs = {n: [] for n in names}
    for n in plan:
        cfg = WORKLOADS[n]
        time.sleep(args.idle)
        runs[n].append(run_one(n, cfg, args.device))

    print(f"\n{'workload':<22} {'quoted':>12} {'value ms':>11} {'CV%':>7} "
          f"{'min':>11} {'max':>11} {'SM MHz':>11} {'n':>3}")
    print("-" * 95)
    out = []
    for n in names:
        rs, cfg = runs[n], WORKLOADS[n]
        # One measurement = one process. `per_launch` turns the span into the
        # per-iteration figure, which for syrk is what one run of the workload
        # actually is.
        div = max(1, rs[0]["launches_per_rep"]) if cfg.get("per_launch") else 1
        v = [r["kernel_ms"] / div for r in rs]
        m = statistics.mean(v)
        cv = 100 * statistics.stdev(v) / m if len(v) > 1 else 0.0
        lo_ms, hi_ms = min(v), max(v)
        clk_lo = min(r["clk_mhz_min"] for r in rs)
        clk_hi = max(r["clk_mhz_mean"] for r in rs)
        print(f"{n:<22} {'kernel_ms':>12} {m:>11.4f} {cv:>6.2f}% {lo_ms:>11.4f} "
              f"{hi_ms:>11.4f} {clk_lo:>4.0f}-{clk_hi:<6.0f} {len(rs):>3}")
        out.append(dict(
            workload=n, shape=rs[0]["shape"], quoted_stat="kernel_ms",
            quoted_ms=round(m, 5), per_launch=bool(cfg.get("per_launch")),
            cv_pct=round(cv, 3), min_ms=round(lo_ms, 5), max_ms=round(hi_ms, 5),
            trials=len(rs), warmup_ms=rs[0]["warmup_ms"],
            clk_mhz_min=clk_lo, clk_mhz_mean=clk_hi,
            boost_mhz=rs[0]["boost_mhz"],
            out_mean=rs[0]["out_mean"], out_std=rs[0]["out_std"],
            why=cfg["note"], runs=rs))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
