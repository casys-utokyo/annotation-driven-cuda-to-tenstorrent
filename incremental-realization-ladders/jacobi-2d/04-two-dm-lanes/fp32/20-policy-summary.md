# Annotations in `jacobi-fp32/unfused-wb-compute-stream-col-2dm`

## What was asked for

| key | first seen | answered by | on | values |
|---|---|---|---|---|
| `tileflow.realization_engine` | `front-2-semantic` | `30-realized-tileflow (before 23-realized-pins.diff)` -- the policy tail, per op (see Support/RealizationEngine.h) | `tileflow.compute_select`, `tileflow.masked_overlay`, `tileflow.row_copy` | `#tileflow.engine<compute>`, `#tileflow.engine<dm>` |
| `tileflow.realization_basis` | `front-2-semantic` | `front-2-conversions` -- overlay-copies / masked-copy-placement | `tileflow.masked_overlay` | `#tileflow.realization_basis<transposed>` |
| `tileflow.conversion_site` | `front-2-conversions` | `30-realized-tileflow (before 23-realized-pins.diff)` -- converted-frame | `tileflow.transpose` | `#tileflow.conversion_site<receiver>`, `#tileflow.conversion_site<sender>` |
| `task.streaming_granularity` | `front-2-semantic` | persists by design -- Physical/SplitFrameSeeds, which writes its default back | `(nested)` | `#task.streaming_granularity<col>` |
| `task.dm_lane` | `30-realized-tileflow` | persists by design -- Tasking/FormTaskGroups -- a HARD constraint, not a request | `tileflow.bulk_transfer`, `tileflow.frame_delivery`, `tileflow.roll` | `#task.engine<dm1>` |

## What the compiler answered

| key | first seen | last seen | values |
|---|---|---|---|
| `tileflow.execution` | `30-realized-tileflow (before 23-realized-pins.diff)` | (to the end) | `"dm"` |
| `tileflow.overlay` | `front-2-conversions` | (to the end) | `#tileflow.overlay_realization<compute_select>`, `#tileflow.overlay_realization<dm_copy>` |
| `task.engine_label` | `0-infer-engines` | (to the end) | `"compute"`, `"dm"` |
| `task.slot_access` | `5-dst-slots` | (to the end) | `#task.slot_access<resident>`, `#task.slot_access<single_pass>` |
| `task.stream` | `50-synchronized-tasks` | (to the end) | `#tileplan.axis<col>` |

