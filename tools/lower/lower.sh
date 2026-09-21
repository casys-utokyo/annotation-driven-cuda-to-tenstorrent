#!/bin/bash
# Rebuild one program from its 70-final-d2m.mlir.
#
#   tools/lower/lower.sh <dir>/70-final-d2m.mlir <outdir>    -> <outdir>/out.ttm
#   FIDELITY=HiFi2 tools/lower/lower.sh <syrk dir>/70-final-d2m.mlir <outdir>
#
# TTMLIR_BIN must point at the tt-mlir fork's build/bin. SYSDESC defaults to the
# system_desc.ttsys at the artifact root.

set -euo pipefail

[ $# -eq 2 ] || { sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 2; }
SRC=$(readlink -f "$1")
mkdir -p "$2"
OUT=$(readlink -f "$2")
ART=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # the artifact root
BIN=${TTMLIR_BIN:?set TTMLIR_BIN to the tt-mlir fork build/bin}
SYSDESC=${SYSDESC:-$ART/system_desc.ttsys}

FIDELITY=${FIDELITY:-HiFi4}
L1ACC=${L1ACC:-false}

echo "[1/3] pass list (fidelity=$FIDELITY, disable-l1-acc=$L1ACC)"
"$BIN/ttmlir-opt" --allow-unregistered-dialect \
  --pass-pipeline="builtin.module( ttcore-register-device{ mock-system-desc-arch=wormhole_b0 system-desc-path=$SYSDESC}, d2m-decompose-arange, d2m-generic-tile-compute-loops{max-dst-physical-size-tiles=0}, d2m-linalg-to-affine{mark-root-loops=true}, d2m-op-scheduler{enable-op-scheduler=true}, d2m-insert-spill-and-scratch, canonicalize, d2m-lower-scratch-allocate, canonicalize, d2m-insert-dst-register-access-unscheduled{disable-l1-acc=$L1ACC max-dst-physical-size-tiles=0}, d2m-insert-dst-register-access-scheduled{disable-l1-acc=$L1ACC max-dst-physical-size-tiles=0}, d2m-insert-tile-matmul-block{use-tile-matmul=false}, d2m-sfpu-tile-loop-fission, canonicalize, func.func( affine-loop-invariant-code-motion ), lower-affine, fold-memref-alias-ops, lower-affine, d2m-generic-linearize-memref, lower-affine, d2m-hoist-cb-allocs, d2m-split-unified-thread, d2m-preallocate-mcast-semaphores, d2m-schedule-dma{num-dm-cores=0}, canonicalize, d2m-lower-load-store-ops-to-dma, d2m-expand-dma-read-composite-view, d2m-lower-dma-to-fully-indexed-form{debug-d2m-coalescing-inference=false}, d2m-normalize-thread-args, d2m-generic-regions-to-funcs, canonicalize, loop-invariant-code-motion, sccp, cse, int-range-optimizations, loop-invariant-code-motion )" \
  "$SRC" -o "$OUT/s1.mlir"

echo "[2/3] d2m -> ttmetal -> emitc"
"$BIN/ttmlir-opt" -allow-unregistered-dialect \
  --convert-d2m-to-ttkernel --canonicalize --ttkernel-control-dst-section \
  --convert-d2m-to-ttmetal=math-fidelity="$FIDELITY" \
  --ttkernel-hoist-inits --convert-ttkernel-to-emitc --canonicalize \
  "$OUT/s1.mlir" -o "$OUT/ttmetal.mlir"

echo "[3/3] translate -> flatbuffer"
"$BIN/ttmlir-translate" --ttmetal-to-flatbuffer -allow-unregistered-dialect \
  - -o "$OUT/out.ttm" < "$OUT/ttmetal.mlir"
echo "OK  $OUT/out.ttm  ($(stat -c %s "$OUT/out.ttm") bytes)"
