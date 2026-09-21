#map = affine_map<(d0, d1) -> ((d1 floordiv 10) * 10 + (d0 floordiv 32) floordiv 16)>
#map1 = affine_map<(d0) -> ((d0 floordiv 32) mod 16)>
#map2 = affine_map<(d0) -> (d0 mod 32)>
#map3 = affine_map<(d0) -> ((d0 floordiv 10) * 10 + d0)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32)>
#map6 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d2 mod 16)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 mod 16)>
#set = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
module {
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute, tt.l1_peak = 671744 : i64, tt.l1_persistent_base = 147456 : i64, tt.l1_pool = 147456 : i64} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg1, %arg2) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(16, 16) order(0) {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tt.frame_pages = 256 : i64, tt.l1_offset = 147456 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 524288 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "persistent"} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<5120x5120xbf16>) {
      ^bb0(%arg3: memref<5120x5120xbf16>):
        %1 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 1) order(0) {task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 0 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 0 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 2048 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 1) order(1) {task.slot_access = #task.slot_access<resident>, tt.cb_id = 1 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 2048 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 2048 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(16, 1) order(2) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 2 : i64, tt.frame_pages = 16 : i64, tt.l1_offset = 4096 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 32768 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 16) order(3) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 3 : i64, tt.frame_pages = 16 : i64, tt.l1_offset = 36864 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 32768 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 16) order(4) {frame_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64, task.slot_access = #task.slot_access<replay>, tt.cb_id = 4 : i64, tt.frame_pages = 16 : i64, tt.l1_offset = 69632 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 32768 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 16) order(5) {frame_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64, task.slot_access = #task.slot_access<replay>, tt.cb_id = 5 : i64, tt.frame_pages = 16 : i64, tt.l1_offset = 102400 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 32768 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(16, 1) resident(1, 1) order(6) {frame_axis = #tileplan.axis<row>, task.drain_axis = #tileplan.axis<row>, task.fill_axis = #tileplan.axis<row>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>, tt.cb_id = 6 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 135168 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(16, 1) resident(1, 1) order(7) {frame_axis = #tileplan.axis<row>, task.drain_axis = #tileplan.axis<row>, task.fill_axis = #tileplan.axis<row>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>, tt.cb_id = 7 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 139264 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<intermediate> name "dst_spill_8" elem bf16 tile(32, 32) domain(1, 1) order(8) {tt.cb_id = 8 : i64, tt.dst_spill, tt.frame_pages = 1 : i64, tt.l1_offset = 143360 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 2048 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<intermediate> name "dst_spill_9" elem bf16 tile(32, 32) domain(1, 1) order(9) {tt.cb_id = 9 : i64, tt.dst_spill, tt.frame_pages = 1 : i64, tt.l1_offset = 145408 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 2048 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<5120x5120xbf16>) writes(%arg3 : memref<5120x5120xbf16>) {producer = "compute", tt.cb_id = 10 : i64}
        %c16 = arith.constant 16 : index
        %c16_1 = arith.constant 16 : index
        tileplan.temporal_for (%arg4) in (0, 5119, 1) {
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes() {
            %11 = affine.apply #map(%arg4, %arg1)
            %12 = affine.apply #map(%arg4, %arg2)
            %13 = affine.apply #map1(%arg4)
            %14 = affine.apply #map1(%arg4)
            %15 = affine.apply #map2(%arg4)
            %16 = affine.apply #map2(%arg4)
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %17 = affine.apply #map2(%arg4)
            %18 = affine.apply #map2(%arg4)
            %19 = affine.apply #map(%arg4, %arg2)
            %20 = affine.apply #map1(%arg4)
            %21 = affine.apply #map(%arg4, %arg1)
            %22 = affine.apply #map1(%arg4)
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %c10_5 = arith.constant 10 : index
            %c10_6 = arith.constant 10 : index
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %1 source_core(%11, %12) source_slice(%13, %14) multicast(%c0_3, %c0_4, %c10_5, %c10_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %c0_7 = arith.constant 0 : index
            task.epoch_wait %1 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %2 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.broadcast_into %1[%c0_7, %c0_7] frag(%15, %16) kind scalar into %2[%c0, %c0_2] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_ready %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %1 epoch(%arg4) : !tileplan.l1_buffer
            %23 = affine.apply #map3(%arg1)
            %24 = affine.apply #map4()
            %c0_8 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c10_9 = arith.constant 10 : index
            tileflow.bulk_transfer %0 into %3 source_core(%23, %19) source_slice(%24, %20) multicast(%arg1, %c0_8, %c1, %c10_9) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %25 = affine.apply #map3(%arg2)
            %26 = affine.apply #map4()
            %c0_10 = arith.constant 0 : index
            %c10_11 = arith.constant 10 : index
            %c1_12 = arith.constant 1 : index
            tileflow.bulk_transfer %0 into %4 source_core(%21, %25) source_slice(%22, %26) multicast(%c0_10, %arg2, %c10_11, %c1_12) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_free %0 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %5 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_13 = arith.constant 0 : index
              %c0_14 = arith.constant 0 : index
              tileflow.broadcast_into %4[%c0_13, %arg5] frag(%18) kind row into %5[%c0_14, %arg5] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            }
            task.epoch_ready %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %6 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_13 = arith.constant 0 : index
              %27 = affine.apply #map5(%arg4, %arg2, %arg5)
              %28 = tileflow.col_mask symbols(%27) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
              tileflow.write_l1_buffer %6 at(%c0_13, %arg5) epoch(%arg4) to %28 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            }
            task.epoch_ready %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %3 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_13 = arith.constant 0 : index
              %c0_14 = arith.constant 0 : index
              task.epoch_acquire %7 epoch(%arg4) : !tileplan.l1_buffer
              tileflow.broadcast_into %3[%arg5, %c0_13] frag(%17) kind col into %7[%arg5, %c0_14] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              task.epoch_ready %7 epoch(%arg4) : !tileplan.l1_buffer
              %c0_15 = arith.constant 0 : index
              %27 = affine.apply #map6(%arg4, %arg1, %arg5)
              %28 = tileflow.row_mask symbols(%27) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
              task.epoch_acquire %8 epoch(%arg4) : !tileplan.l1_buffer
              tileflow.write_l1_buffer %8 at(%arg5, %c0_15) epoch(%arg4) to %28 : !tileplan.l1_buffer to !tileflow.mask<bf16>
              task.epoch_ready %8 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %3 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_1 engine(<compute>) domain(<phase>) reads() writes() {
            %11 = affine.apply #map2(%arg4)
            %12 = affine.apply #map2(%arg4)
            %13 = affine.apply #map2(%arg4)
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            task.epoch_acquire %9 epoch(%arg4) : !tileplan.l1_buffer
            %14 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
            task.epoch_wait %2 epoch(%arg4) : !tileplan.l1_buffer
            %15 = tileflow.read_l1_buffer %2 at(%c0, %c0_2) on cell(%11, %12) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
            %16 = d2mtile.unary tile_recip %15 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
            task.epoch_free %2 epoch(%arg4) : !tileplan.l1_buffer
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            tileflow.write_l1_buffer %9 at(%c0_3, %c0_4) epoch(%arg4) to %16 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
            task.epoch_ready %9 epoch(%arg4) : !tileplan.l1_buffer
            d2mtile.dst_release %14 : !d2mtile.dst_bank
            task.epoch_acquire %0 epoch(%arg4) {task.prime_before_loop} : !tileplan.l1_buffer
            task.epoch_wait %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %9 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_5 = arith.constant 0 : index
              task.epoch_wait %7 epoch(%arg4) : !tileplan.l1_buffer
              %17 = tileflow.read_l1_buffer %7 at(%arg5, %c0_5) on col(%13) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %c0_6 = arith.constant 0 : index
              %c0_7 = arith.constant 0 : index
              %18 = tileflow.read_l1_buffer %9 at(%c0_6, %c0_7) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              task.epoch_acquire %10 epoch(%arg4) : !tileplan.l1_buffer
              %19 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
              %20 = d2mtile.binary tile_mul %17, %18 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
              task.epoch_free %7 epoch(%arg4) : !tileplan.l1_buffer
              %c0_8 = arith.constant 0 : index
              %c0_9 = arith.constant 0 : index
              tileflow.write_l1_buffer %10 at(%c0_8, %c0_9) epoch(%arg4) to %20 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              task.epoch_ready %10 epoch(%arg4) : !tileplan.l1_buffer
              d2mtile.dst_release %19 : !d2mtile.dst_bank
              %c0_10 = arith.constant 0 : index
              task.epoch_wait %8 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_wait %10 epoch(%arg4) : !tileplan.l1_buffer
              tileplan.shard_for (%arg6) in (%c16_1) axis <col> {
                %c0_11 = arith.constant 0 : index
                %21 = tileflow.read_l1_buffer %5 at(%c0_11, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %22 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                %23 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                %24 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %25 = tileflow.read_l1_buffer %0 at(%22, %23) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %c0_12 = arith.constant 0 : index
                %c0_13 = arith.constant 0 : index
                %26 = tileflow.read_l1_buffer %10 at(%c0_12, %c0_13) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %27 = d2mtile.binary tile_mul %26, %21 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                %28 = d2mtile.binary tile_sub %25, %27 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                %29 = tileflow.read_l1_buffer %8 at(%arg5, %c0_10) epoch(%arg4) {tt.dst_slot = 2 : i64} : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %30 = tileflow.compute_select %28 with %25 active(%29 : !tileflow.mask<bf16>) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>, tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                %c0_14 = arith.constant 0 : index
                %31 = tileflow.read_l1_buffer %6 at(%c0_14, %arg6) epoch(%arg4) {tt.dst_slot = 2 : i64} : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %32 = tileflow.compute_select %30 with %25 active(%31 : !tileflow.mask<bf16>) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>, tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %32 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                d2mtile.dst_release %24 : !d2mtile.dst_bank
              }
              task.epoch_free %10 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_free %8 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %9 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_ready %0 epoch(%arg4) : !tileplan.l1_buffer
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

