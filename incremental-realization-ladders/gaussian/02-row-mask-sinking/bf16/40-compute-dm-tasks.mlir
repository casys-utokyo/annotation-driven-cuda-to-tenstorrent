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
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(16, 16) order(0) {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16]} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<5120x5120xbf16>) {
      ^bb0(%arg3: memref<5120x5120xbf16>):
        %1 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 1) order(0) : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 1) order(1) : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(16, 1) order(2) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 16) order(3) {frame_axis = #tileplan.axis<col>} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 16) order(4) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 16) order(5) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 16 : i64} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(16, 1) order(6) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(16, 1) order(7) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<5120x5120xbf16>) writes(%arg3 : memref<5120x5120xbf16>) {producer = "compute"}
        tileplan.temporal_for (%arg4) in (0, 5119, 1) {
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes(%1 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full]) {
            %c16 = arith.constant 16 : index
            %c16_1 = arith.constant 16 : index
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c10_3 = arith.constant 10 : index
            %c10_4 = arith.constant 10 : index
            %c0_5 = arith.constant 0 : index
            %9 = affine.apply #map(%arg1)
            %10 = affine.apply #map1()
            %c0_6 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c10_7 = arith.constant 10 : index
            %11 = affine.apply #map(%arg2)
            %12 = affine.apply #map1()
            %c0_8 = arith.constant 0 : index
            %c10_9 = arith.constant 10 : index
            %c1_10 = arith.constant 1 : index
            %c0_11 = arith.constant 0 : index
            %c0_12 = arith.constant 0 : index
            %13 = affine.apply #map2(%arg4, %arg1)
            %14 = affine.apply #map2(%arg4, %arg2)
            %15 = affine.apply #map3(%arg4)
            %16 = affine.apply #map3(%arg4)
            tileflow.bulk_transfer %0 into %1 source_core(%13, %14) source_slice(%15, %16) multicast(%c0, %c0_2, %c10_3, %c10_4) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %17 = affine.apply #map4(%arg4)
            %18 = affine.apply #map4(%arg4)
            %c0_13 = arith.constant 0 : index
            %c0_14 = arith.constant 0 : index
            tileflow.broadcast_into %1[%c0_5, %c0_5] frag(%17, %18) kind scalar into %2[%c0_13, %c0_14] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %19 = affine.apply #map2(%arg4, %arg2)
            %20 = affine.apply #map3(%arg4)
            tileflow.bulk_transfer %0 into %3 source_core(%9, %19) source_slice(%10, %20) multicast(%arg1, %c0_6, %c1, %c10_7) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %21 = affine.apply #map2(%arg4, %arg1)
            %22 = affine.apply #map3(%arg4)
            tileflow.bulk_transfer %0 into %4 source_core(%21, %11) source_slice(%22, %12) multicast(%c0_8, %arg2, %c10_9, %c1_10) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %23 = affine.apply #map4(%arg4)
              %c0_15 = arith.constant 0 : index
              tileflow.broadcast_into %4[%c0_12, %arg5] frag(%23) kind row into %5[%c0_15, %arg5] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            }
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_15 = arith.constant 0 : index
              %23 = affine.apply #map5(%arg4, %arg2, %arg5)
              %24 = tileflow.col_mask symbols(%23) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
              tileflow.write_l1_buffer %6 at(%c0_15, %arg5) epoch(%arg4) to %24 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            }
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %23 = affine.apply #map4(%arg4)
              %c0_15 = arith.constant 0 : index
              tileflow.broadcast_into %3[%arg5, %c0_11] frag(%23) kind col into %7[%arg5, %c0_15] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              %c0_16 = arith.constant 0 : index
              %24 = affine.apply #map6(%arg4, %arg1, %arg5)
              %25 = tileflow.row_mask symbols(%24) set #set {task.engine_label = "dm"} : !tileflow.mask<bf16>
              tileflow.write_l1_buffer %8 at(%arg5, %c0_16) epoch(%arg4) to %25 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            }
          }
          task.group @compute_1 engine(<compute>) domain(<phase>) reads(%2 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %0 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full]) writes(%0 : !tileplan.l1_buffer [full]) {
            %c16 = arith.constant 16 : index
            %c16_1 = arith.constant 16 : index
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %9 = d2mtile.fill "0.000000e+00" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %10 = affine.apply #map4(%arg4)
            %11 = affine.apply #map4(%arg4)
            %12 = tileflow.read_l1_buffer %2 at(%c0_3, %c0_4) on cell(%10, %11) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
            %13 = d2mtile.unary tile_recip %12 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_5 = arith.constant 0 : index
              %14 = affine.apply #map4(%arg4)
              %15 = tileflow.read_l1_buffer %7 at(%arg5, %c0_5) on col(%14) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %c0_6 = arith.constant 0 : index
              %16 = tileflow.read_l1_buffer %8 at(%arg5, %c0_6) epoch(%arg4) : !tileplan.l1_buffer -> !tileflow.mask<bf16>
              %17 = tileflow.compute_select %9 with %15 active(%16 : !tileflow.mask<bf16>) {task.engine_label = "compute"} : !d2mtile.tile<bf16>
              %18 = d2mtile.binary tile_mul %17, %13 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
              tileplan.shard_for (%arg6) in (%c16_1) axis <col> {
                %19 = tileflow.read_l1_buffer %5 at(%c0, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %20 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                %21 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                %22 = tileflow.read_l1_buffer %0 at(%20, %21) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %23 = d2mtile.binary tile_mul %18, %19 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %24 = d2mtile.binary tile_sub %22, %23 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %25 = tileflow.read_l1_buffer %6 at(%c0_2, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %26 = tileflow.compute_select %24 with %22 active(%25 : !tileflow.mask<bf16>) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %26 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

