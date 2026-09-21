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
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg1, %arg2) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(16, 16) order(0) {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16]} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<5120x5120xbf16>) {
      ^bb0(%arg3: memref<5120x5120xbf16>):
        %1 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 1) order(0) {task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 1) order(1) {task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(16, 1) order(2) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 16) order(3) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 16) order(4) {frame_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64, task.slot_access = #task.slot_access<replay>} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(16, 1) resident(1, 1) order(5) {frame_axis = #tileplan.axis<row>, task.drain_axis = #tileplan.axis<row>, task.fill_axis = #tileplan.axis<row>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<intermediate> name "dst_spill_6" elem bf16 tile(32, 32) domain(1, 1) order(6) {tt.dst_spill} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<intermediate> name "dst_spill_7" elem bf16 tile(32, 32) domain(1, 1) order(7) {tt.dst_spill} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<5120x5120xbf16>) writes(%arg3 : memref<5120x5120xbf16>) {producer = "compute"}
        %c16 = arith.constant 16 : index
        %c16_1 = arith.constant 16 : index
        tileplan.temporal_for (%arg4) in (0, 5119, 1) {
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes() {
            %9 = affine.apply #map(%arg4, %arg1)
            %10 = affine.apply #map(%arg4, %arg2)
            %11 = affine.apply #map1(%arg4)
            %12 = affine.apply #map1(%arg4)
            %13 = affine.apply #map2(%arg4)
            %14 = affine.apply #map2(%arg4)
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %15 = affine.apply #map2(%arg4)
            %16 = affine.apply #map2(%arg4)
            %17 = affine.apply #map(%arg4, %arg2)
            %18 = affine.apply #map1(%arg4)
            %19 = affine.apply #map(%arg4, %arg1)
            %20 = affine.apply #map1(%arg4)
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %c10_5 = arith.constant 10 : index
            %c10_6 = arith.constant 10 : index
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %1 source_core(%9, %10) source_slice(%11, %12) multicast(%c0_3, %c0_4, %c10_5, %c10_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %c0_7 = arith.constant 0 : index
            task.epoch_wait %1 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %2 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.broadcast_into %1[%c0_7, %c0_7] frag(%13, %14) kind scalar into %2[%c0, %c0_2] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_ready %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %1 epoch(%arg4) : !tileplan.l1_buffer
            %21 = affine.apply #map3(%arg1)
            %22 = affine.apply #map4()
            %c0_8 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c10_9 = arith.constant 10 : index
            tileflow.bulk_transfer %0 into %3 source_core(%21, %17) source_slice(%22, %18) multicast(%arg1, %c0_8, %c1, %c10_9) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %23 = affine.apply #map3(%arg2)
            %24 = affine.apply #map4()
            %c0_10 = arith.constant 0 : index
            %c10_11 = arith.constant 10 : index
            %c1_12 = arith.constant 1 : index
            tileflow.bulk_transfer %0 into %4 source_core(%19, %23) source_slice(%20, %24) multicast(%c0_10, %arg2, %c10_11, %c1_12) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_free %0 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %5 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_13 = arith.constant 0 : index
              %c32 = arith.constant 32 : index
              %c1_14 = arith.constant 1 : index
              %25 = affine.apply #map5(%arg4, %arg2, %arg5)
              %c0_15 = arith.constant 0 : index
              scf.for %arg6 = %c0_13 to %c32 step %c1_14 {
                %c0_16 = arith.constant 0 : index
                %26 = tileflow.element_load %4[%c0_16, %arg5] at(%16, %arg6) {task.engine_label = "dm"} : !tileplan.l1_buffer -> bf16
                %c0_17 = arith.constant 0 : index
                %27 = affine.apply #map6(%arg6)[%25]
                %28 = arith.cmpi sge, %27, %c0_17 : index
                %cst = arith.constant 0.000000e+00 : bf16
                %29 = arith.select %28, %26, %cst : bf16
                tileflow.element_store %29 to %5[%c0_15, %arg5] at(%16, %arg6) {task.engine_label = "dm"} : bf16 to !tileplan.l1_buffer
              } {task.engine_label = "dm"}
              tileflow.broadcast_into %5[%c0_15, %arg5] frag(%16) kind row into %5[%c0_15, %arg5] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            }
            task.epoch_ready %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %3 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_13 = arith.constant 0 : index
              %c32 = arith.constant 32 : index
              %c1_14 = arith.constant 1 : index
              %25 = affine.apply #map7(%arg4, %arg1, %arg5)
              %c0_15 = arith.constant 0 : index
              task.epoch_acquire %6 epoch(%arg4) : !tileplan.l1_buffer
              scf.for %arg6 = %c0_13 to %c32 step %c1_14 {
                %c0_16 = arith.constant 0 : index
                %26 = tileflow.element_load %3[%arg5, %c0_16] at(%arg6, %15) {task.engine_label = "dm"} : !tileplan.l1_buffer -> bf16
                %c0_17 = arith.constant 0 : index
                %27 = affine.apply #map6(%arg6)[%25]
                %28 = arith.cmpi sge, %27, %c0_17 : index
                %cst = arith.constant 0.000000e+00 : bf16
                %29 = arith.select %28, %26, %cst : bf16
                tileflow.element_store %29 to %6[%arg5, %c0_15] at(%arg6, %15) {task.engine_label = "dm"} : bf16 to !tileplan.l1_buffer
              } {task.engine_label = "dm"}
              tileflow.broadcast_into %6[%arg5, %c0_15] frag(%15) kind col into %6[%arg5, %c0_15] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              task.epoch_ready %6 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %3 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_0 engine(<compute>) domain(<phase>) reads() writes() {
            %9 = affine.apply #map2(%arg4)
            %10 = affine.apply #map2(%arg4)
            %11 = affine.apply #map2(%arg4)
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            task.epoch_acquire %7 epoch(%arg4) : !tileplan.l1_buffer
            %12 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
            task.epoch_wait %2 epoch(%arg4) : !tileplan.l1_buffer
            %13 = tileflow.read_l1_buffer %2 at(%c0, %c0_2) on cell(%9, %10) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
            %14 = d2mtile.unary tile_recip %13 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
            task.epoch_free %2 epoch(%arg4) : !tileplan.l1_buffer
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            tileflow.write_l1_buffer %7 at(%c0_3, %c0_4) epoch(%arg4) to %14 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
            task.epoch_ready %7 epoch(%arg4) : !tileplan.l1_buffer
            d2mtile.dst_release %12 : !d2mtile.dst_bank
            task.epoch_acquire %0 epoch(%arg4) {task.prime_before_loop} : !tileplan.l1_buffer
            task.epoch_wait %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %7 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_5 = arith.constant 0 : index
              task.epoch_wait %6 epoch(%arg4) : !tileplan.l1_buffer
              %15 = tileflow.read_l1_buffer %6 at(%arg5, %c0_5) on col(%11) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %c0_6 = arith.constant 0 : index
              %c0_7 = arith.constant 0 : index
              %16 = tileflow.read_l1_buffer %7 at(%c0_6, %c0_7) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              task.epoch_acquire %8 epoch(%arg4) : !tileplan.l1_buffer
              %17 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
              %18 = d2mtile.binary tile_mul %15, %16 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
              task.epoch_free %6 epoch(%arg4) : !tileplan.l1_buffer
              %c0_8 = arith.constant 0 : index
              %c0_9 = arith.constant 0 : index
              tileflow.write_l1_buffer %8 at(%c0_8, %c0_9) epoch(%arg4) to %18 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              task.epoch_ready %8 epoch(%arg4) : !tileplan.l1_buffer
              d2mtile.dst_release %17 : !d2mtile.dst_bank
              task.epoch_wait %8 epoch(%arg4) : !tileplan.l1_buffer
              tileplan.shard_for (%arg6) in (%c16_1) axis <col> {
                %19 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                %20 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                %21 = tileflow.read_l1_buffer %0 at(%19, %20) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %c0_10 = arith.constant 0 : index
                %22 = tileflow.read_l1_buffer %5 at(%c0_10, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %c0_11 = arith.constant 0 : index
                %c0_12 = arith.constant 0 : index
                %23 = tileflow.read_l1_buffer %8 at(%c0_11, %c0_12) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %24 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %25 = d2mtile.binary tile_mul %23, %22 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                %26 = d2mtile.binary tile_sub %21, %25 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %26 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                d2mtile.dst_release %24 : !d2mtile.dst_bank
              }
              task.epoch_free %8 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %7 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_ready %0 epoch(%arg4) : !tileplan.l1_buffer
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

