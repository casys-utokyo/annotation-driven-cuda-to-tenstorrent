#!/bin/sh
# How this directory was lowered, written out as the commands that did it.
#
#   TOTT=<to-tt> TASK_OPT=<task-opt> ./REPRO.sh [outdir]   # default ./repro
#   TOTT=... TASK_OPT=... ./REPRO.sh .                     # in place, for `git diff`
#
# 80-program.ttm is built separately, by tools/lower/lower.sh.
set -eu
TOTT=${TOTT:?set TOTT to the paper compiler, to-tt}
TASK_OPT=${TASK_OPT:?set TASK_OPT to task-opt}
OUT=${1:-./repro}
mkdir -p "$OUT"

run() {  # <input> <output> <to-tt flag>...
  src=$1; dst=$2; shift 2
  "$TOTT" "$@" "$src" > "$dst.raw"
  "$TASK_OPT" "$dst.raw" > "$dst"
  rm -f "$dst.raw"
}

# --- the annotated front -------------------------------------------------
BASE=${BASE:-../../../../shared-inputs/gaussian-fp32}
W="$OUT/.authoring"
cp "$BASE/20-semantic-tileflow.mlir" "$W"
patch -s -p0 "$W" < 21-policy-annotations.diff
run "$W" "$W.next" --apply-tileflow-policy \
    --apply-tileflow-policy-stop-after=predicate-sinking
mv "$W.next" "$W"
"$TASK_OPT" "$W" > "$OUT/22-annotated-tileflow.mlir"

# --- realize the policy ---------------------------------------------------
run "$OUT/22-annotated-tileflow.mlir" "$OUT/30-realized-tileflow.mlir" --realize-tileflow-policy

# --- the task pipeline, each stage from the realized front ----------------
FRONT="$OUT/30-realized-tileflow.mlir"
run "$FRONT" "$OUT/40-compute-dm-tasks.mlir" --form-task-groups
run "$FRONT" "$OUT/50-synchronized-tasks.mlir" --realize-epoch-sync
run "$FRONT" "$OUT/51-buffer-depth.mlir" --realize-epoch-depth
run "$FRONT" "$OUT/60-resource-bindings.mlir" --bind-phase-cbs
"$TOTT" --emit-d2m "$FRONT" > "$OUT/70-final-d2m.mlir"   # task-opt cannot parse D2M

# --- what moved ----------------------------------------------------------
[ "$OUT" = "." ] && exit 0
bad=0
for f in 22-annotated-tileflow.mlir \
         30-realized-tileflow.mlir \
         40-compute-dm-tasks.mlir \
         50-synchronized-tasks.mlir \
         51-buffer-depth.mlir \
         60-resource-bindings.mlir \
         70-final-d2m.mlir; do
  if cmp -s "$OUT/$f" "$f"; then
    echo "  $f  same"
  else
    echo "  $f  DIFFERS  (diff -u $f $OUT/$f)"
    bad=$((bad + 1))
  fi
done
rm -f "$OUT/.authoring"
[ "$bad" = 0 ] || { echo "$bad file(s) differ from what is shipped here"; exit 1; }
echo "all checkpoints reproduced"
