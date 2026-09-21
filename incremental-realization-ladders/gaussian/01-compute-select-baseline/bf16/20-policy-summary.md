# Annotations in `gaussian/recip-compute`

## What was asked for

| key | first seen | answered by | on | values |
|---|---|---|---|---|
| `task.streaming_granularity` | `22-annotated-tileflow` | persists by design -- Physical/SplitFrameSeeds, which writes its default back | `(nested)` | `#task.streaming_granularity<row>` |

## What the compiler answered

| key | first seen | last seen | values |
|---|---|---|---|
| `tileflow.overlay` | `30-realized-tileflow` | (to the end) | `#tileflow.overlay_realization<compute_select>` |
| `task.engine_label` | `0-infer-engines` | (to the end) | `"compute"`, `"dm"` |
| `task.slot_access` | `50-synchronized-tasks` | (to the end) | `#task.slot_access<replay>`, `#task.slot_access<resident>`, `#task.slot_access<single_pass>` |
| `task.stream` | `50-synchronized-tasks` | (to the end) | `#tileplan.axis<row>` |

