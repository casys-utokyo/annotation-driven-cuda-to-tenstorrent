#map = affine_map<(d0) -> ((d0 floordiv 10) * 10 + d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0, d1) -> ((d1 floordiv 10) * 10 + (d0 floordiv 32) floordiv 16)>
#map3 = affine_map<(d0) -> ((d0 floordiv 32) mod 16)>
#map4 = affine_map<(d0) -> (d0 mod 32)>
#map5 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32)>
#map6 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d2 mod 16)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 mod 16)>
#set = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
module {
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg1, %arg2) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(16, 16) order(0) {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<5120x5120xbf16>) {
      ^bb0(%arg3: memref<5120x5120xbf16>):
        %1 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 1) order(0) {task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 1) order(1) {task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(16, 1) order(2) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 16) order(3) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 16) order(4) {frame_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64, task.slot_access = #task.slot_access<replay>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 16) order(5) {frame_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64, task.slot_access = #task.slot_access<replay>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(16, 1) resident(1, 1) order(6) {frame_axis = #tileplan.axis<row>, task.drain_axis = #tileplan.axis<row>, task.fill_axis = #tileplan.axis<row>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(16, 1) resident(1, 1) order(7) {frame_axis = #tileplan.axis<row>, task.drain_axis = #tileplan.axis<row>, task.fill_axis = #tileplan.axis<row>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<intermediate> name "dst_spill_8" elem bf16 tile(32, 32) domain(1, 1) order(8) {tt.dst_spill, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<intermediate> name "dst_spill_9" elem bf16 tile(32, 32) domain(1, 1) order(9) {tt.dst_spill, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %11 = tileplan.l1_buffer role<intermediate> name "dst_spill_10" elem bf16 tile(32, 32) domain(1, 1) order(10) {tt.dst_spill, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<5120x5120xbf16>) writes(%arg3 : memref<5120x5120xbf16>) {producer = "compute"}
        tileplan.temporal_for (%arg4) in (0, 5119, 1) {
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes() {
            %c16 = arith.constant 16 : index
            %c16_1 = arith.constant 16 : index
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c10_3 = arith.constant 10 : index
            %c10_4 = arith.constant 10 : index
            %c0_5 = arith.constant 0 : index
            %12 = affine.apply #map(%arg1)
            %13 = affine.apply #map1()
            %c0_6 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c10_7 = arith.constant 10 : index
            %14 = affine.apply #map(%arg2)
            %15 = affine.apply #map1()
            %c0_8 = arith.constant 0 : index
            %c10_9 = arith.constant 10 : index
            %c1_10 = arith.constant 1 : index
            %c0_11 = arith.constant 0 : index
            %c0_12 = arith.constant 0 : index
            %16 = affine.apply #map2(%arg4, %arg1)
            %17 = affine.apply #map2(%arg4, %arg2)
            %18 = affine.apply #map3(%arg4)
            %19 = affine.apply #map3(%arg4)
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %1 source_core(%16, %17) source_slice(%18, %19) multicast(%c0, %c0_2, %c10_3, %c10_4) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %20 = affine.apply #map4(%arg4)
            %21 = affine.apply #map4(%arg4)
            %c0_13 = arith.constant 0 : index
            %c0_14 = arith.constant 0 : index
            task.epoch_wait %1 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %2 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.broadcast_into %1[%c0_5, %c0_5] frag(%20, %21) kind scalar into %2[%c0_13, %c0_14] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_ready %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %1 epoch(%arg4) : !tileplan.l1_buffer
            %22 = affine.apply #map2(%arg4, %arg2)
            %23 = affine.apply #map3(%arg4)
            tileflow.bulk_transfer %0 into %3 source_core(%12, %22) source_slice(%13, %23) multicast(%arg1, %c0_6, %c1, %c10_7) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %24 = affine.apply #map2(%arg4, %arg1)
            %25 = affine.apply #map3(%arg4)
            tileflow.bulk_transfer %0 into %4 source_core(%24, %14) source_slice(%25, %15) multicast(%c0_8, %arg2, %c10_9, %c1_10) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_free %0 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %5 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %26 = affine.apply #map4(%arg4)
              %c0_15 = arith.constant 0 : index
              tileflow.broadcast_into %4[%c0_12, %arg5] frag(%26) kind row into %5[%c0_15, %arg5] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            }
            task.epoch_ready %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %6 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_15 = arith.constant 0 : index
              %26 = affine.apply #map5(%arg4, %arg2, %arg5)
              %27 = tileflow.col_mask symbols(%26) set #set {task.engine_label = "dm"} : !tileflow.mask<bf16>
              tileflow.write_l1_buffer %6 at(%c0_15, %arg5) epoch(%arg4) to %27 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            }
            task.epoch_ready %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %3 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %26 = affine.apply #map4(%arg4)
              %c0_15 = arith.constant 0 : index
              task.epoch_acquire %7 epoch(%arg4) : !tileplan.l1_buffer
              tileflow.broadcast_into %3[%arg5, %c0_11] frag(%26) kind col into %7[%arg5, %c0_15] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              task.epoch_ready %7 epoch(%arg4) : !tileplan.l1_buffer
              %c0_16 = arith.constant 0 : index
              %27 = affine.apply #map6(%arg4, %arg1, %arg5)
              %28 = tileflow.row_mask symbols(%27) set #set {task.engine_label = "dm"} : !tileflow.mask<bf16>
              task.epoch_acquire %8 epoch(%arg4) : !tileplan.l1_buffer
              tileflow.write_l1_buffer %8 at(%arg5, %c0_16) epoch(%arg4) to %28 : !tileplan.l1_buffer to !tileflow.mask<bf16>
              task.epoch_ready %8 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %3 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_1 engine(<compute>) domain(<phase>) reads() writes() {
            %c16 = arith.constant 16 : index
            %c16_1 = arith.constant 16 : index
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            task.epoch_acquire %9 epoch(%arg4) : !tileplan.l1_buffer
            %12 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
            %13 = d2mtile.fill "0.000000e+00" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            tileflow.write_l1_buffer %9 at(%c0_3, %c0_4) epoch(%arg4) to %13 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
            task.epoch_ready %9 epoch(%arg4) : !tileplan.l1_buffer
            d2mtile.dst_release %12 : !d2mtile.dst_bank
            task.epoch_acquire %10 epoch(%arg4) : !tileplan.l1_buffer
            %14 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
            %c0_5 = arith.constant 0 : index
            %c0_6 = arith.constant 0 : index
            %15 = affine.apply #map4(%arg4)
            %16 = affine.apply #map4(%arg4)
            task.epoch_wait %2 epoch(%arg4) : !tileplan.l1_buffer
            %17 = tileflow.read_l1_buffer %2 at(%c0_5, %c0_6) on cell(%15, %16) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
            %18 = d2mtile.unary tile_recip %17 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
            task.epoch_free %2 epoch(%arg4) : !tileplan.l1_buffer
            %c0_7 = arith.constant 0 : index
            %c0_8 = arith.constant 0 : index
            tileflow.write_l1_buffer %10 at(%c0_7, %c0_8) epoch(%arg4) to %18 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
            task.epoch_ready %10 epoch(%arg4) : !tileplan.l1_buffer
            d2mtile.dst_release %14 : !d2mtile.dst_bank
            task.epoch_acquire %0 epoch(%arg4) {task.prime_before_loop} : !tileplan.l1_buffer
            task.epoch_wait %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %9 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %10 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_9 = arith.constant 0 : index
              %19 = affine.apply #map4(%arg4)
              task.epoch_acquire %11 epoch(%arg4) : !tileplan.l1_buffer
              %20 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
              task.epoch_wait %7 epoch(%arg4) : !tileplan.l1_buffer
              %21 = tileflow.read_l1_buffer %7 at(%arg5, %c0_9) on col(%19) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %c0_10 = arith.constant 0 : index
              task.epoch_wait %8 epoch(%arg4) : !tileplan.l1_buffer
              %22 = tileflow.read_l1_buffer %8 at(%arg5, %c0_10) epoch(%arg4) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !tileflow.mask<bf16>
              %c0_11 = arith.constant 0 : index
              %c0_12 = arith.constant 0 : index
              %23 = tileflow.read_l1_buffer %9 at(%c0_11, %c0_12) epoch(%arg4) {tt.dst_slot = 2 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %24 = tileflow.compute_select %23 with %21 active(%22 : !tileflow.mask<bf16>) {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
              task.epoch_free %8 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_free %7 epoch(%arg4) : !tileplan.l1_buffer
              %c0_13 = arith.constant 0 : index
              %c0_14 = arith.constant 0 : index
              %25 = tileflow.read_l1_buffer %10 at(%c0_13, %c0_14) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %26 = d2mtile.binary tile_mul %24, %25 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
              %c0_15 = arith.constant 0 : index
              %c0_16 = arith.constant 0 : index
              tileflow.write_l1_buffer %11 at(%c0_15, %c0_16) epoch(%arg4) to %26 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              task.epoch_ready %11 epoch(%arg4) : !tileplan.l1_buffer
              d2mtile.dst_release %20 : !d2mtile.dst_bank
              task.epoch_wait %11 epoch(%arg4) : !tileplan.l1_buffer
              tileplan.shard_for (%arg6) in (%c16_1) axis <col> {
                %27 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %28 = tileflow.read_l1_buffer %5 at(%c0, %arg6) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %29 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                %30 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                %31 = tileflow.read_l1_buffer %0 at(%29, %30) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %32 = tileflow.read_l1_buffer %6 at(%c0_2, %arg6) epoch(%arg4) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %c0_17 = arith.constant 0 : index
                %c0_18 = arith.constant 0 : index
                %33 = tileflow.read_l1_buffer %9 at(%c0_17, %c0_18) epoch(%arg4) {tt.dst_slot = 2 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %34 = tileflow.compute_select %33 with %28 active(%32 : !tileflow.mask<bf16>) {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                %c0_19 = arith.constant 0 : index
                %c0_20 = arith.constant 0 : index
                %35 = tileflow.read_l1_buffer %11 at(%c0_19, %c0_20) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %36 = d2mtile.binary tile_mul %35, %34 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                %37 = d2mtile.binary tile_sub %31, %36 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %37 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                d2mtile.dst_release %27 : !d2mtile.dst_bank
              }
              task.epoch_free %11 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %10 epoch(%arg4) : !tileplan.l1_buffer
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

