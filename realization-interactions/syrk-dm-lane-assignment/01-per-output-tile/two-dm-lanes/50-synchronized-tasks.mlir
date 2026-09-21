#map = affine_map<(d0, d1, d2) -> ((d2 floordiv 10) * 10 + d1 floordiv 10 + d0)>
#map1 = affine_map<(d0, d1) -> ((d1 floordiv 10) * 10 + d0 floordiv 10)>
#map2 = affine_map<(d0) -> (d0 mod 10)>
#map3 = affine_map<(d0, d1) -> ((d0 floordiv 10) * 10 + d1 floordiv 10 + d0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 mod 10)>
module {
  func.func @_Z11syrk_kerneliiPfS__driver(%arg0: memref<3200x3200xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<input>}, %arg1: memref<3200x3200xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(10, 10) order(0) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10]} : !tileplan.l1_buffer
      %1 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(10, 10) order(1) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10]} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins(%arg0 : memref<3200x3200xbf16>) outs(%arg1 : memref<3200x3200xbf16>) {
      ^bb0(%arg4: memref<3200x3200xbf16>, %arg5: memref<3200x3200xbf16>):
        %2 = tileplan.l1_buffer role<intermediate> elem bf16 tile(32, 32) domain(1, 1) order(2) {tt.contraction_accumulator} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 1) order(0) {task.buffer_depth = 2 : i64, task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 1) order(1) {task.buffer_depth = 2 : i64, task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<intermediate> name "dst_spill_3" elem bf16 tile(32, 32) domain(1, 1) order(3) {tt.dst_spill} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<intermediate> name "dst_spill_4" elem bf16 tile(32, 32) domain(1, 1) order(4) {tt.dst_spill} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg4 : memref<3200x3200xbf16>) writes()
        tileplan.bind_persistent %1 : !tileplan.l1_buffer reads(%arg5 : memref<3200x3200xbf16>) writes(%arg5 : memref<3200x3200xbf16>) {producer = "compute"}
        task.group @dm1_0 engine(<dm1>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes() {
          %c10_1 = arith.constant 10 : index
          %c10_2 = arith.constant 10 : index
          %c0 = arith.constant 0 : index
          %c10_3 = arith.constant 10 : index
          %c1 = arith.constant 1 : index
          tileplan.shard_for (%arg6) in (%c10_1) axis <row> {
            tileplan.shard_for (%arg7) in (%c10_2) axis <col> {
              tileplan.temporal_for (%arg8) in (0, 100, 1) {
                %7 = affine.apply #map(%arg3, %arg7, %arg2)
                %8 = affine.apply #map1(%arg8, %arg3)
                %9 = affine.apply #map2(%arg7)
                %10 = affine.apply #map2(%arg8)
                tileflow.bulk_transfer %0 into %3 source_core(%7, %8) source_slice(%9, %10) multicast(%c0, %arg3, %c10_3, %c1) epoch(%arg8) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              } {epoch_id = 0 : i64}
            }
          }
        }
        task.group @dm0_0 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes() {
          %c10_1 = arith.constant 10 : index
          %c10_2 = arith.constant 10 : index
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c10_3 = arith.constant 10 : index
          tileplan.shard_for (%arg6) in (%c10_1) axis <row> {
            tileplan.shard_for (%arg7) in (%c10_2) axis <col> {
              tileplan.temporal_for (%arg8) in (0, 100, 1) {
                %7 = affine.apply #map3(%arg2, %arg6)
                %8 = affine.apply #map1(%arg8, %arg3)
                %9 = affine.apply #map2(%arg6)
                %10 = affine.apply #map2(%arg8)
                tileflow.bulk_transfer %0 into %4 source_core(%7, %8) source_slice(%9, %10) multicast(%arg2, %c0, %c1, %c10_3) epoch(%arg8) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              } {epoch_id = 0 : i64}
            }
          }
        }
        task.group @compute_1 engine(<compute>) domain(<phase>) reads(%1 : !tileplan.l1_buffer [full]) writes(%1 : !tileplan.l1_buffer [full]) {
          %c10_1 = arith.constant 10 : index
          %c10_2 = arith.constant 10 : index
          %c0 = arith.constant 0 : index
          %c0_3 = arith.constant 0 : index
          task.epoch_acquire %5 : !tileplan.l1_buffer
          %7 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
          %8 = d2mtile.fill "3.000000e+00" -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
          %c0_4 = arith.constant 0 : index
          %c0_5 = arith.constant 0 : index
          tileflow.write_l1_buffer %5 at(%c0_4, %c0_5) to %8 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
          task.epoch_ready %5 : !tileplan.l1_buffer
          task.epoch_wait %5 : !tileplan.l1_buffer
          d2mtile.dst_release %7 : !d2mtile.dst_bank
          task.epoch_acquire %6 : !tileplan.l1_buffer
          %9 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
          %10 = d2mtile.fill "2.000000e+00" -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
          %c0_6 = arith.constant 0 : index
          %c0_7 = arith.constant 0 : index
          tileflow.write_l1_buffer %6 at(%c0_6, %c0_7) to %10 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
          task.epoch_ready %6 : !tileplan.l1_buffer
          task.epoch_wait %6 : !tileplan.l1_buffer
          d2mtile.dst_release %9 : !d2mtile.dst_bank
          task.epoch_acquire %2 : !tileplan.l1_buffer
          task.epoch_acquire %1 : !tileplan.l1_buffer
          tileplan.shard_for (%arg6) in (%c10_1) axis <row> {
            tileplan.shard_for (%arg7) in (%c10_2) axis <col> {
              %11 = affine.apply #map4(%arg2, %arg3, %arg6, %arg7)
              %12 = affine.apply #map5(%arg2, %arg3, %arg6, %arg7)
              %13 = tileflow.read_l1_buffer %1 at(%11, %12) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %14 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
              %15 = d2mtile.fill "0.000000e+00" -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
              %16 = tileplan.temporal_for (%arg8) in (0, 100, 1) iter_args(%arg9 = %15) -> (!d2mtile.tile<bf16>) {
                task.epoch_wait %3 epoch(%arg8) : !tileplan.l1_buffer
                %22 = tileflow.read_l1_buffer %3 at(%c0, %c0) epoch(%arg8) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                task.epoch_wait %4 epoch(%arg8) : !tileplan.l1_buffer
                %23 = tileflow.read_l1_buffer %4 at(%c0_3, %c0_3) epoch(%arg8) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %24 = d2mtile.matmul_block %23, %22 acc %arg9 block [1, 1, 1] {task.engine_label = "compute", transpose = true, tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %4 epoch(%arg8) : !tileplan.l1_buffer
                task.epoch_free %3 epoch(%arg8) : !tileplan.l1_buffer
                %c0_12 = arith.constant 0 : index
                tileflow.write_l1_buffer %2 at(%c0_12, %c0_12) to %24 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                tileplan.yield %24 : !d2mtile.tile<bf16>
              } {epoch_id = 0 : i64, tt.dst_slot = 0 : i64}
              %c0_8 = arith.constant 0 : index
              %c0_9 = arith.constant 0 : index
              %17 = tileflow.read_l1_buffer %5 at(%c0_8, %c0_9) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %18 = d2mtile.binary tile_mul %13, %17 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
              %c0_10 = arith.constant 0 : index
              %c0_11 = arith.constant 0 : index
              %19 = tileflow.read_l1_buffer %6 at(%c0_10, %c0_11) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %20 = d2mtile.binary tile_mul %16, %19 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
              %21 = d2mtile.binary tile_add %18, %20 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
              tileflow.write_l1_buffer %1 at(%arg6, %arg7) to %21 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              d2mtile.dst_release %14 : !d2mtile.dst_bank
            }
          }
          task.epoch_ready %1 : !tileplan.l1_buffer
          task.epoch_free %6 : !tileplan.l1_buffer
          task.epoch_free %5 : !tileplan.l1_buffer
        }
      } {task.streaming_granularity = #task.streaming_granularity<frame>}
    }
    return
  }
}

