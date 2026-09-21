# Annotations in `gaussian/mask-dm-recip-compute`

## What was asked for

| key | first seen | answered by | on | values |
|---|---|---|---|---|
| `tileflow.realization_engine` | `front-2-annotated` | `30-realized-tileflow` -- the policy tail, per op (see Support/RealizationEngine.h) | `d2mtile.unary`, `tileflow.mask_fill` | `#tileflow.engine<compute>`, `#tileflow.engine<dm>` |
| `tileflow.predication` | `front-2-annotated` | `front-2-predicated` -- predicate-sinking | `tileflow.masked_overlay` | (unit) |
| `tileflow.predication_target` | `front-2-annotated` | `front-2-predicated` -- predicate-sinking | `tileflow.broadcast` | `0`, `1` |
| `task.streaming_granularity` | `front-2-annotated` | persists by design -- Physical/SplitFrameSeeds, which writes its default back | `(nested)` | `#task.streaming_granularity<row>` |

## What the compiler answered

| key | first seen | last seen | values |
|---|---|---|---|
| `tileflow.execution` | `30-realized-tileflow` | `0-infer-engines` | `"dm"` |
| `task.engine_label` | `0-infer-engines` | (to the end) | `"compute"`, `"dm"` |
| `task.slot_access` | `50-synchronized-tasks` | (to the end) | `#task.slot_access<replay>`, `#task.slot_access<resident>`, `#task.slot_access<single_pass>` |
| `task.stream` | `50-synchronized-tasks` | (to the end) | `#tileplan.axis<row>` |

