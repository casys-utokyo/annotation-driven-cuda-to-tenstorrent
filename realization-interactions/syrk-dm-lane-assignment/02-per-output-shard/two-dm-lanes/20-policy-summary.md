# Annotations in `syrk/kexpand-2dm`

## What was asked for

| key | first seen | answered by | on | values |
|---|---|---|---|---|
| `task.buffer_depth` | `30-realized-tileflow` | persists by design -- Tasking/EmitLogicalL1Buffers -> RealizeEpochDepth | `tileflow.frame_delivery`, `tileplan.l1_buffer` | `2` |
| `task.dm_lane` | `30-realized-tileflow` | persists by design -- Tasking/FormTaskGroups -- a HARD constraint, not a request | `tileflow.bulk_transfer`, `tileflow.frame_delivery` | `#task.engine<dm1>` |

## What the compiler answered

| key | first seen | last seen | values |
|---|---|---|---|
| `task.streaming_granularity` | `50-synchronized-tasks` | (to the end) | `#task.streaming_granularity<frame>` |
| `task.engine_label` | `0-infer-engines` | (to the end) | `"compute"`, `"dm"` |
| `task.slot_access` | `50-synchronized-tasks` | (to the end) | `#task.slot_access<resident>` |

