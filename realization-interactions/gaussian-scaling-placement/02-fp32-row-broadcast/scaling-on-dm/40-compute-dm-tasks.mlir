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
        %6 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(16, 1) order(5) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<5120x5120xf32>) writes(%arg3 : memref<5120x5120xf32>) {producer = "compute"}
        %c16 = arith.constant 16 : index
        %c16_1 = arith.constant 16 : index
        tileplan.temporal_for (%arg4) in (0, 5119, 1) {
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes(%1 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full]) {
            %7 = affine.apply #map(%arg4, %arg1)
            %8 = affine.apply #map(%arg4, %arg2)
            %9 = affine.apply #map1(%arg4)
            %10 = affine.apply #map1(%arg4)
            %11 = affine.apply #map2(%arg4)
            %12 = affine.apply #map2(%arg4)
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %cst = arith.constant 1.000000e+00 : f32
            %13 = affine.apply #map2(%arg4)
            %14 = affine.apply #map2(%arg4)
            %15 = affine.apply #map(%arg4, %arg2)
            %16 = affine.apply #map1(%arg4)
            %17 = affine.apply #map(%arg4, %arg1)
            %18 = affine.apply #map1(%arg4)
            %c0_5 = arith.constant 0 : index
            %c0_6 = arith.constant 0 : index
            %c10_7 = arith.constant 10 : index
            %c10_8 = arith.constant 10 : index
            tileflow.bulk_transfer %0 into %1 source_core(%7, %8) source_slice(%9, %10) multicast(%c0_5, %c0_6, %c10_7, %c10_8) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %19 = tileflow.element_load %1[%c0_3, %c0_4] at(%11, %12) {task.engine_label = "dm"} : !tileplan.l1_buffer -> f32
            %20 = arith.divf %cst, %19 : f32
            tileflow.element_store %20 to %2[%c0, %c0_2] at(%11, %12) {task.engine_label = "dm"} : f32 to !tileplan.l1_buffer
            %21 = affine.apply #map3(%arg1)
            %22 = affine.apply #map4()
            %c0_9 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c10_10 = arith.constant 10 : index
            tileflow.bulk_transfer %0 into %3 source_core(%21, %15) source_slice(%22, %16) multicast(%arg1, %c0_9, %c1, %c10_10) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            %23 = affine.apply #map3(%arg2)
            %24 = affine.apply #map4()
            %c0_11 = arith.constant 0 : index
            %c10_12 = arith.constant 10 : index
            %c1_13 = arith.constant 1 : index
            tileflow.bulk_transfer %0 into %4 source_core(%17, %23) source_slice(%18, %24) multicast(%c0_11, %arg2, %c10_12, %c1_13) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c16_1) axis <col> {
              %c0_14 = arith.constant 0 : index
              %c32 = arith.constant 32 : index
              %c1_15 = arith.constant 1 : index
              %25 = affine.apply #map5(%arg4, %arg2, %arg5)
              %c0_16 = arith.constant 0 : index
              scf.for %arg6 = %c0_14 to %c32 step %c1_15 {
                %c0_17 = arith.constant 0 : index
                %26 = tileflow.element_load %4[%c0_17, %arg5] at(%14, %arg6) {task.engine_label = "dm"} : !tileplan.l1_buffer -> f32
                %c0_18 = arith.constant 0 : index
                %27 = affine.apply #map6(%arg6)[%25]
                %28 = arith.cmpi sge, %27, %c0_18 : index
                %cst_19 = arith.constant 0.000000e+00 : f32
                %29 = arith.select %28, %26, %cst_19 : f32
                tileflow.element_store %29 to %5[%c0_16, %arg5] at(%c0_14, %arg6) {task.engine_label = "dm"} : f32 to !tileplan.l1_buffer
              } {task.engine_label = "dm"}
            }
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0_14 = arith.constant 0 : index
              %c32 = arith.constant 32 : index
              %c1_15 = arith.constant 1 : index
              %25 = affine.apply #map7(%arg4, %arg1, %arg5)
              %c0_16 = arith.constant 0 : index
              scf.for %arg6 = %c0_14 to %c32 step %c1_15 {
                %c0_17 = arith.constant 0 : index
                %26 = tileflow.element_load %3[%arg5, %c0_17] at(%arg6, %13) {task.engine_label = "dm"} : !tileplan.l1_buffer -> f32
                %c0_18 = arith.constant 0 : index
                %27 = affine.apply #map6(%arg6)[%25]
                %28 = arith.cmpi sge, %27, %c0_18 : index
                %cst_19 = arith.constant 0.000000e+00 : f32
                %29 = arith.select %28, %26, %cst_19 : f32
                %c0_20 = arith.constant 0 : index
                %c0_21 = arith.constant 0 : index
                %30 = tileflow.element_load %2[%c0_20, %c0_21] at(%11, %12) {task.engine_label = "dm"} : !tileplan.l1_buffer -> f32
                %31 = arith.mulf %29, %30 : f32
                tileflow.element_store %31 to %6[%arg5, %c0_16] at(%arg6, %13) {task.engine_label = "dm"} : f32 to !tileplan.l1_buffer
              } {task.engine_label = "dm"}
              tileflow.broadcast_into %6[%arg5, %c0_16] frag(%13) kind col into %6[%arg5, %c0_16] {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            }
          }
          task.group @compute_0_1 engine(<compute>) domain(<phase>) reads(%6 : !tileplan.l1_buffer [full], %0 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full]) writes(%0 : !tileplan.l1_buffer [full]) {
            %7 = affine.apply #map2(%arg4)
            tileplan.shard_for (%arg5) in (%c16) axis <row> {
              %c0 = arith.constant 0 : index
              %8 = tileflow.read_l1_buffer %6 at(%arg5, %c0) on col(%7) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
              tileplan.shard_for (%arg6) in (%c16_1) axis <col> {
                %9 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                %10 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                %11 = tileflow.read_l1_buffer %0 at(%9, %10) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %c0_2 = arith.constant 0 : index
                %12 = tileflow.read_l1_buffer %5 at(%c0_2, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %13 = d2mtile.binary tile_mul %8, %12 -> "source_ssa" bcast row {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %14 = d2mtile.binary tile_sub %11, %13 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %14 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

