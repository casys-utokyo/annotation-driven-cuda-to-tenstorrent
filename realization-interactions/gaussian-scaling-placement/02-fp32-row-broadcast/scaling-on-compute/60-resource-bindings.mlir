#map = affine_map<(d0, d1) -> ((d1 floordiv 10) * 10 + (d0 floordiv 32) floordiv 16)>
#map1 = affine_map<(d0) -> ((d0 floordiv 32) mod 16)>
#map2 = affine_map<(d0) -> (d0 mod 32)>
#map3 = affine_map<(d0) -> ((d0 floordiv 10) * 10 + d0)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32)>
#map6 = affine_map<(d0)[s0] -> (d0 + s0)>
#map7 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 mod 16)>
module {
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xf32> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute, tt.l1_peak = 1265664 : i64, tt.l1_persistent_base = 217088 : i64, tt.l1_pool = 217088 : i64} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg1, %arg2) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem f32 tile(32, 32) domain(16, 16) order(0) {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tt.frame_pages = 256 : i64, tt.l1_offset = 217088 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 1048576 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "persistent"} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<5120x5120xf32>) {
      ^bb0(%arg3: memref<5120x5120xf32>):
        %1 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(1, 1) order(0) {task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 0 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 0 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(1, 1) order(1) {task.slot_access = #task.slot_access<resident>, tt.cb_id = 1 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 4096 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(16, 1) order(2) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 2 : i64, tt.frame_pages = 16 : i64, tt.l1_offset = 8192 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 65536 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(1, 16) order(3) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 3 : i64, tt.frame_pages = 16 : i64, tt.l1_offset = 73728 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 65536 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(1, 16) order(4) {frame_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64, task.slot_access = #task.slot_access<replay>, tt.cb_id = 4 : i64, tt.frame_pages = 16 : i64, tt.l1_offset = 139264 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 65536 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(16, 1) resident(1, 1) order(5) {frame_axis = #tileplan.axis<row>, task.drain_axis = #tileplan.axis<row>, task.fill_axis = #tileplan.axis<row>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>, tt.cb_id = 5 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 204800 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 8192 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<intermediate> name "dst_spill_6" elem f32 tile(32, 32) domain(1, 1) order(6) {tt.cb_id = 6 : i64, tt.dst_spill, tt.frame_pages = 1 : i64, tt.l1_offset = 212992 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<5120x5120xf32>) writes(%arg3 : memref<5120x5120xf32>) {producer = "compute", tt.cb_id = 7 : i64}
        %c16 = arith.constant 16 : index
        %c16_1 = arith.constant 16 : index
        tileplan.temporal_for (%arg4) in (0, 5119, 1) {
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes() {
            %8 = affine.apply #map(%arg4, %arg1)
            %9 = affine.apply #map(%arg4, %arg2)
            %10 = affine.apply #map1(%arg4)
            %11 = affine.apply #map1(%arg4)
            %12 = affine.apply #map2(%arg4)
            %13 = affine.apply #map2(%arg4)
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %cst = arith.constant 1.000000e+00 : f32
            %14 = affine.apply #map2(%arg4)
            %15 = affine.apply #map2(%arg4)
            %16 = affine.apply #map(%arg4, %arg2)
            %17 = affine.apply #map1(%arg4)
            %18 = affine.apply #map(%arg4, %arg1)
            %19 = affine.apply #map1(%arg4)
            %c0_5 = arith.constant 0 : index
            %c0_6 = arith.constant 0 : index
            %c10_7 = arith.constant 10 : index
            %c10_8 = arith.constant 10 : index
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %1 source_core(%8, %9) source_slice(%10, %11) multicast(%c0_5, %c0_6, %c10_7, %c10_8) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_wait %1 epoch(%arg4) : !tileplan.l1_buffer
            %20 = tileflow.element_load %1[%c0_3, %c0_4] at(%12, %13) {task.engine_label = "dm"} : !tileplan.l1_buffer -> f32
            task.epoch_free %1 epoch(%arg4) : !tileplan.l1_buffer
            %21 = arith.divf %cst, %20 : f32
            task.epoch_acquire %2 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.element_store %21 to %2[%c0, %c0_2] at(%12, %13) {task.engine_label = "dm"} : f32 to !tileplan.l1_buffer
            tileflow.broadcast_into %2[%c0, %c0_2] frag(%12, %13) kind scalar into %2[%c0, %c0_2] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_ready %2 epoch(%arg4) : !tileplan.l1_buffer
            %22 = affine.apply #map3(%arg1)
            %23 = affine.apply #map4()
            %c0_9 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c10_10 = arith.constant 10 : index
            tileflow.bulk_transfer %0 into %3 source_core(%22, %16) source_slice(%23, %17) multicast(%arg1, %c0_9, %c1, %c10_10) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %24 = affine.apply #map3(%arg2)
            %25 = affine.apply #map4()
            %c0_11 = arith.constant 0 : index
            %c10_12 = arith.constant 10 : index
            %c1_13 = arith.constant 1 : index
            tileflow.bulk_transfer %0 into %4 source_core(%18, %24) source_slice(%19, %25) multicast(%c0_11, %arg2, %c10_12, %c1_13) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_free %0 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %5 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_14 = arith.constant 0 : index
              %c32 = arith.constant 32 : index
              %c1_15 = arith.constant 1 : index
              %26 = affine.apply #map5(%arg4, %arg2, %arg5)
              %c0_16 = arith.constant 0 : index
              scf.for %arg6 = %c0_14 to %c32 step %c1_15 {
                %c0_17 = arith.constant 0 : index
                %27 = tileflow.element_load %4[%c0_17, %arg5] at(%15, %arg6) {task.engine_label = "dm"} : !tileplan.l1_buffer -> f32
                %c0_18 = arith.constant 0 : index
                %28 = affine.apply #map6(%arg6)[%26]
                %29 = arith.cmpi sge, %28, %c0_18 : index
                %cst_19 = arith.constant 0.000000e+00 : f32
                %30 = arith.select %29, %27, %cst_19 : f32
                tileflow.element_store %30 to %5[%c0_16, %arg5] at(%c0_14, %arg6) {task.engine_label = "dm"} : f32 to !tileplan.l1_buffer
              } {task.engine_label = "dm"}
            }
            task.epoch_ready %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %3 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_14 = arith.constant 0 : index
              %c32 = arith.constant 32 : index
              %c1_15 = arith.constant 1 : index
              %26 = affine.apply #map7(%arg4, %arg1, %arg5)
              %c0_16 = arith.constant 0 : index
              task.epoch_acquire %6 epoch(%arg4) : !tileplan.l1_buffer
              scf.for %arg6 = %c0_14 to %c32 step %c1_15 {
                %c0_17 = arith.constant 0 : index
                %27 = tileflow.element_load %3[%arg5, %c0_17] at(%arg6, %14) {task.engine_label = "dm"} : !tileplan.l1_buffer -> f32
                %c0_18 = arith.constant 0 : index
                %28 = affine.apply #map6(%arg6)[%26]
                %29 = arith.cmpi sge, %28, %c0_18 : index
                %cst_19 = arith.constant 0.000000e+00 : f32
                %30 = arith.select %29, %27, %cst_19 : f32
                tileflow.element_store %30 to %6[%arg5, %c0_16] at(%arg6, %14) {task.engine_label = "dm"} : f32 to !tileplan.l1_buffer
              } {task.engine_label = "dm"}
              tileflow.broadcast_into %6[%arg5, %c0_16] frag(%14) kind col into %6[%arg5, %c0_16] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              task.epoch_ready %6 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %3 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_0 engine(<compute>) domain(<phase>) reads() writes() {
            %8 = affine.apply #map2(%arg4)
            %9 = affine.apply #map2(%arg4)
            %10 = affine.apply #map2(%arg4)
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            task.epoch_acquire %0 epoch(%arg4) {task.prime_before_loop} : !tileplan.l1_buffer
            task.epoch_wait %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %5 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_3 = arith.constant 0 : index
              task.epoch_acquire %7 epoch(%arg4) : !tileplan.l1_buffer
              %11 = d2mtile.dst_acquire capacity 4 : !d2mtile.dst_bank
              task.epoch_wait %6 epoch(%arg4) : !tileplan.l1_buffer
              %12 = tileflow.read_l1_buffer %6 at(%arg5, %c0_3) on col(%10) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
              %13 = tileflow.read_l1_buffer %2 at(%c0, %c0_2) on cell(%8, %9) epoch(%arg4) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
              %14 = d2mtile.binary tile_mul %12, %13 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<f32>
              task.epoch_free %6 epoch(%arg4) : !tileplan.l1_buffer
              %c0_4 = arith.constant 0 : index
              %c0_5 = arith.constant 0 : index
              tileflow.write_l1_buffer %7 at(%c0_4, %c0_5) epoch(%arg4) to %14 : !tileplan.l1_buffer to !d2mtile.tile<f32>
              task.epoch_ready %7 epoch(%arg4) : !tileplan.l1_buffer
              d2mtile.dst_release %11 : !d2mtile.dst_bank
              task.epoch_wait %7 epoch(%arg4) : !tileplan.l1_buffer
              tileplan.shard_for (%arg6) in (%c16_1) axis <col> {
                %15 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                %16 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                %17 = d2mtile.dst_acquire capacity 4 : !d2mtile.dst_bank
                %18 = tileflow.read_l1_buffer %0 at(%15, %16) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %c0_6 = arith.constant 0 : index
                %19 = tileflow.read_l1_buffer %5 at(%c0_6, %arg6) epoch(%arg4) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %c0_7 = arith.constant 0 : index
                %c0_8 = arith.constant 0 : index
                %20 = tileflow.read_l1_buffer %7 at(%c0_7, %c0_8) epoch(%arg4) {tt.dst_slot = 2 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %21 = d2mtile.binary tile_mul %20, %19 -> "source_ssa" bcast row {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<f32>
                %22 = d2mtile.binary tile_sub %18, %21 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %22 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
                d2mtile.dst_release %17 : !d2mtile.dst_bank
              }
              task.epoch_free %7 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_ready %0 epoch(%arg4) : !tileplan.l1_buffer
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

