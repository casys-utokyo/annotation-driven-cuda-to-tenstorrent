# Annotations in `syrk/unblocked`

## What was asked for

No loop-schedule annotation: the reduction is left as the source writes it, one tile per step. The plan is `15-scheduled-tileplan.mlir`.

## What the compiler answered

| key | first seen | last seen | values |
|---|---|---|---|
| `task.streaming_granularity` | `50-synchronized-tasks` | (to the end) | `#task.streaming_granularity<frame>` |
| `task.engine_label` | `0-infer-engines` | (to the end) | `"compute"`, `"dm"` |
| `task.slot_access` | `50-synchronized-tasks` | (to the end) | `#task.slot_access<resident>` |

