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
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xf32> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg1, %arg2) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem f32 tile(32, 32) domain(16, 16) order(0) {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16]} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<5120x5120xf32>) {
      ^bb0(%arg3: memref<5120x5120xf32>):
        %1 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(1, 1) order(0) : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(1, 1) order(1) : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(16, 1) order(2) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(1, 16) order(3) {frame_axis = #tileplan.axis<col>} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(1, 16) order(4) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(1, 16) order(5) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(16, 1) order(6) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(16, 1) order(7) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<5120x5120xf32>) writes(%arg3 : memref<5120x5120xf32>) {producer = "compute"}
        %c16 = arith.constant 16 : index
        %c16_1 = arith.constant 16 : index
        tileplan.temporal_for (%arg4) in (0, 5119, 1) {
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes(%1 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full]) {
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
            tileflow.bulk_transfer %0 into %1 source_core(%9, %10) source_slice(%11, %12) multicast(%c0_3, %c0_4, %c10_5, %c10_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %c0_7 = arith.constant 0 : index
            tileflow.broadcast_into %1[%c0_7, %c0_7] frag(%13, %14) kind scalar into %2[%c0, %c0_2] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
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
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_13 = arith.constant 0 : index
              %c0_14 = arith.constant 0 : index
              tileflow.broadcast_into %4[%c0_13, %arg5] frag(%16) kind row into %5[%c0_14, %arg5] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            }
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_13 = arith.constant 0 : index
              %25 = affine.apply #map5(%arg4, %arg2, %arg5)
              %26 = tileflow.col_mask symbols(%25) set #set invert {task.engine_label = "dm"} : !tileflow.mask<f32>
              tileflow.write_l1_buffer %6 at(%c0_13, %arg5) epoch(%arg4) to %26 : !tileplan.l1_buffer to !tileflow.mask<f32>
            }
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_13 = arith.constant 0 : index
              %c0_14 = arith.constant 0 : index
              tileflow.broadcast_into %3[%arg5, %c0_13] frag(%15) kind col into %7[%arg5, %c0_14] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              %c0_15 = arith.constant 0 : index
              %25 = affine.apply #map6(%arg4, %arg1, %arg5)
              %26 = tileflow.row_mask symbols(%25) set #set invert {task.engine_label = "dm"} : !tileflow.mask<f32>
              tileflow.write_l1_buffer %8 at(%arg5, %c0_15) epoch(%arg4) to %26 : !tileplan.l1_buffer to !tileflow.mask<f32>
            }
          }
          task.group @compute_1 engine(<compute>) domain(<phase>) reads(%2 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %0 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full]) writes(%0 : !tileplan.l1_buffer [full]) {
            %9 = affine.apply #map2(%arg4)
            %10 = affine.apply #map2(%arg4)
            %11 = affine.apply #map2(%arg4)
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %12 = tileflow.read_l1_buffer %2 at(%c0, %c0_2) on cell(%9, %10) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
            %13 = d2mtile.unary tile_recip %12 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_3 = arith.constant 0 : index
              %14 = tileflow.read_l1_buffer %7 at(%arg5, %c0_3) on col(%11) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
              %15 = d2mtile.binary tile_mul %14, %13 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
              %c0_4 = arith.constant 0 : index
              %16 = tileflow.read_l1_buffer %8 at(%arg5, %c0_4) epoch(%arg4) : !tileplan.l1_buffer -> !tileflow.mask<f32>
              tileplan.shard_for (%arg6) in (%c16_1) axis <col> {
                %c0_5 = arith.constant 0 : index
                %17 = tileflow.read_l1_buffer %5 at(%c0_5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %18 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                %19 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                %20 = tileflow.read_l1_buffer %0 at(%18, %19) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %21 = d2mtile.binary tile_mul %15, %17 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %22 = d2mtile.binary tile_sub %20, %21 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %23 = tileflow.compute_select %22 with %20 active(%16 : !tileflow.mask<f32>) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
                %c0_6 = arith.constant 0 : index
                %24 = tileflow.read_l1_buffer %6 at(%c0_6, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !tileflow.mask<f32>
                %25 = tileflow.compute_select %23 with %20 active(%24 : !tileflow.mask<f32>) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %25 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

