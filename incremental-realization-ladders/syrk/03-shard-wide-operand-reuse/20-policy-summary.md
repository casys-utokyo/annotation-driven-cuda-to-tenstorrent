# Annotations in `syrk/kexpand`

## What was asked for

A loop-schedule annotation, applied in `21-schedule.attr`:

    [#tileplan.strip_mine<loop = "k", factor = 10>, #tileplan.reorder<writes_abi = 1, nest = [#tileplan.temporal<"k">, #tileplan.axis<row>, #tileplan.axis<col>]>]

The scheduled plan is `15-scheduled-tileplan.mlir`.

## What the compiler answered

| key | first seen | last seen | values |
|---|---|---|---|
| `task.streaming_granularity` | `50-synchronized-tasks` | (to the end) | `#task.streaming_granularity<frame>` |
| `task.engine_label` | `0-infer-engines` | (to the end) | `"compute"`, `"dm"` |
| `task.slot_access` | `50-synchronized-tasks` | (to the end) | `#task.slot_access<resident>` |

