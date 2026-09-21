# Annotations in `jacobi/unfused-wb-compute-receive-transpose`

## What was asked for

| key | first seen | answered by | on | values |
|---|---|---|---|---|
| `task.streaming_granularity` | `22-annotated-tileflow` | persists by design -- Physical/SplitFrameSeeds, which writes its default back | `(nested)` | `#task.streaming_granularity<col>` |

## What the compiler answered

| key | first seen | last seen | values |
|---|---|---|---|
| `tileflow.execution` | `30-realized-tileflow` | (to the end) | `"dm"` |
| `tileflow.overlay` | `30-realized-tileflow` | (to the end) | `#tileflow.overlay_realization<compute_select>`, `#tileflow.overlay_realization<dm_copy>` |
| `task.engine_label` | `0-infer-engines` | (to the end) | `"compute"`, `"dm"` |
| `task.slot_access` | `5-dst-slots` | (to the end) | `#task.slot_access<resident>`, `#task.slot_access<single_pass>` |
| `task.stream` | `50-synchronized-tasks` | (to the end) | `#tileplan.axis<col>` |

