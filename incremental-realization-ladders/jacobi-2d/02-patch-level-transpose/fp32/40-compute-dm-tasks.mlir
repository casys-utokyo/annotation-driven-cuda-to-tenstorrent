#map = affine_map<(d0, d1) -> ((d0 * 10 + d1) * 32 - 1)>
#map1 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * -32 + 3198)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<(d0) -> (d0 mod 10)>
#map4 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 - 1) mod 10)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 mod 10)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10)>
#map7 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 + 1) mod 10)>
#map8 = affine_map<(d0, d1, d2, d3) -> ((d3 + d1 * 10 + 1) mod 10)>
#map9 = affine_map<(d0, d1, d2, d3) -> ((d3 + d1 * 10 - 1) mod 10)>
#map10 = affine_map<()[s0] -> (s0)>
#map11 = affine_map<()[s0] -> (s0 - 31)>
#set = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 >= 0)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xf32> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg1, %arg2) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem f32 tile(32, 32) domain(10, 10) order(0) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10]} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<3200x3200xf32>) {
      ^bb0(%arg3: memref<3200x3200xf32>):
        %1 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(0) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(2, 1) order(1) {frame_axis = #tileplan.axis<row>, task.frame_access = "replay", task.reuse = 1000 : i64} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(1, 2) order(2) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 10 : i64} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(1, 10) order(3) {frame_axis = #tileplan.axis<col>} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(1, 10) order(4) {frame_axis = #tileplan.axis<col>} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(10, 1) order(5) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(10, 1) order(6) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(7) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(8) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(9) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %11 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(10) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %12 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(11) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %13 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(12) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %14 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(13) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %15 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(14) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<3200x3200xf32>) writes(%arg3 : memref<3200x3200xf32>) {producer = "dm"}
        task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes(%2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) {
          %c1 = arith.constant 1 : index
          %c1_1 = arith.constant 1 : index
          tileplan.shard_for (%arg4) in (%c1) axis <row> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %16 = affine.apply #map(%arg1, %c0_3)
            %17 = tileflow.row_mask symbols(%16) set #set invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %2 at(%c0, %c0_2) to %17 : !tileplan.l1_buffer to !tileflow.mask<f32>
            %c1_4 = arith.constant 1 : index
            %c0_5 = arith.constant 0 : index
            %c9 = arith.constant 9 : index
            %18 = affine.apply #map1(%arg1, %c9)
            %19 = tileflow.row_mask symbols(%18) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %2 at(%c1_4, %c0_5) to %19 : !tileplan.l1_buffer to !tileflow.mask<f32>
          }
          tileplan.shard_for (%arg4) in (%c1_1) axis <col> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %16 = affine.apply #map(%arg2, %c0_3)
            %17 = tileflow.col_mask symbols(%16) set #set invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %3 at(%c0, %c0_2) to %17 : !tileplan.l1_buffer to !tileflow.mask<f32>
            %c0_4 = arith.constant 0 : index
            %c1_5 = arith.constant 1 : index
            %c9 = arith.constant 9 : index
            %18 = affine.apply #map1(%arg2, %c9)
            %19 = tileflow.col_mask symbols(%18) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %3 at(%c0_4, %c1_5) to %19 : !tileplan.l1_buffer to !tileflow.mask<f32>
          }
        }
        tileplan.temporal_for (%arg4) in (0, 1000, 1) {
          task.group @dm0_0_1 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full], %10 : !tileplan.l1_buffer [full], %11 : !tileplan.l1_buffer [full], %13 : !tileplan.l1_buffer [full], %14 : !tileplan.l1_buffer [full]) writes(%4 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %9 : !tileplan.l1_buffer [full], %12 : !tileplan.l1_buffer [full], %15 : !tileplan.l1_buffer [full]) {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %true = arith.constant true
            %true_3 = arith.constant true
            %16 = arith.andi %true, %true_3 : i1
            %true_4 = arith.constant true
            %true_5 = arith.constant true
            %17 = arith.andi %true_4, %true_5 : i1
            %c-1 = arith.constant -1 : index
            %18 = arith.addi %arg1, %c-1 : index
            %c0 = arith.constant 0 : index
            %19 = arith.maxsi %18, %c0 : index
            %c9 = arith.constant 9 : index
            %c0_6 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %20 = arith.addi %arg1, %c1 : index
            %c9_7 = arith.constant 9 : index
            %21 = arith.minsi %20, %c9_7 : index
            %c0_8 = arith.constant 0 : index
            %c0_9 = arith.constant 0 : index
            %c1_10 = arith.constant 1 : index
            %22 = arith.addi %arg2, %c1_10 : index
            %c9_11 = arith.constant 9 : index
            %23 = arith.minsi %22, %c9_11 : index
            %c0_12 = arith.constant 0 : index
            %c0_13 = arith.constant 0 : index
            %c-1_14 = arith.constant -1 : index
            %24 = arith.addi %arg2, %c-1_14 : index
            %c0_15 = arith.constant 0 : index
            %25 = arith.maxsi %24, %c0_15 : index
            %c0_16 = arith.constant 0 : index
            %c9_17 = arith.constant 9 : index
            %c0_18 = arith.constant 0 : index
            %c1_19 = arith.constant 1 : index
            %c0_20 = arith.constant 0 : index
            %c10_21 = arith.constant 10 : index
            %26 = affine.apply #map2()
            %c9_22 = arith.constant 9 : index
            %c10_23 = arith.constant 10 : index
            %c0_24 = arith.constant 0 : index
            %c10_25 = arith.constant 10 : index
            %27 = affine.apply #map2()
            %true_26 = arith.constant true
            %true_27 = arith.constant true
            %28 = arith.andi %true_26, %true_27 : i1
            %true_28 = arith.constant true
            %true_29 = arith.constant true
            %29 = arith.andi %true_28, %true_29 : i1
            %true_30 = arith.constant true
            %true_31 = arith.constant true
            %true_32 = arith.constant true
            %true_33 = arith.constant true
            tileflow.bulk_transfer %0 into %4 source_core(%19, %arg2) source_slice(%c9, %c0_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %5 source_core(%21, %arg2) source_slice(%c0_8, %c0_9) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %6 source_core(%arg1, %23) source_slice(%c0_12, %c0_13) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %7 source_core(%arg1, %25) source_slice(%c0_16, %c9_17) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %30 = arith.cmpi sge, %arg5, %c0_18 : index
                %31 = arith.cmpi slt, %arg5, %c1_19 : index
                %32 = arith.andi %30, %31 : i1
                %33 = arith.cmpi sge, %arg6, %c0_20 : index
                %34 = arith.cmpi slt, %arg6, %c10_21 : index
                %35 = arith.andi %33, %34 : i1
                %36 = arith.andi %32, %35 : i1
                %37 = affine.apply #map3(%arg6)
                %38 = tileflow.read_l1_buffer %4 at(%26, %37) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %39 = scf.if %36 -> (!d2mtile.tile<f32>) {
                  scf.yield %38 : !d2mtile.tile<f32>
                } else {
                  %74 = affine.apply #map4(%arg1, %arg2, %arg5, %arg6)
                  %75 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %76 = tileflow.read_l1_buffer %0 at(%74, %75) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %76 : !d2mtile.tile<f32>
                } {task.engine_label = "dm"}
                %40 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %41 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %42 = tileflow.read_l1_buffer %0 at(%40, %41) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %43 = tileflow.roll %42 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %44 = tileflow.lane_copy %43 with %42 if(%true_30) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                %45 = tileflow.lane_copy %44 with %39 if(%16) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) to %45 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %46 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %47 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %48 = tileflow.read_l1_buffer %0 at(%46, %47) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %49 = arith.cmpi sge, %arg5, %c9_22 : index
                %50 = arith.cmpi slt, %arg5, %c10_23 : index
                %51 = arith.andi %49, %50 : i1
                %52 = arith.cmpi sge, %arg6, %c0_24 : index
                %53 = arith.cmpi slt, %arg6, %c10_25 : index
                %54 = arith.andi %52, %53 : i1
                %55 = arith.andi %51, %54 : i1
                %56 = affine.apply #map3(%arg6)
                %57 = tileflow.read_l1_buffer %5 at(%27, %56) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %58 = scf.if %55 -> (!d2mtile.tile<f32>) {
                  scf.yield %57 : !d2mtile.tile<f32>
                } else {
                  %74 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                  %75 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %76 = tileflow.read_l1_buffer %0 at(%74, %75) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %76 : !d2mtile.tile<f32>
                } {task.engine_label = "dm"}
                %59 = tileflow.roll %48 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %60 = tileflow.lane_copy %59 with %48 if(%true_31) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                %61 = tileflow.lane_copy %60 with %58 if(%17) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) to %61 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %62 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %63 = tileflow.roll %62 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %64 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %65 = tileflow.lane_copy %63 with %64 if(%true_32) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                %66 = tileflow.read_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %67 = tileflow.lane_copy %65 with %66 if(%28) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) to %67 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %68 = tileflow.read_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %69 = tileflow.roll %68 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %70 = tileflow.read_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %71 = tileflow.lane_copy %69 with %70 if(%true_33) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                %72 = tileflow.read_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %73 = tileflow.lane_copy %71 with %72 if(%29) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %15 at(%arg5, %arg6) epoch(%arg4) to %73 : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
          }
          task.group @compute_0 engine(<compute>) domain(<phase>) reads(%4 : !tileplan.l1_buffer [full], %0 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %12 : !tileplan.l1_buffer [full], %15 : !tileplan.l1_buffer [full], %9 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) writes(%10 : !tileplan.l1_buffer [full], %11 : !tileplan.l1_buffer [full], %13 : !tileplan.l1_buffer [full], %14 : !tileplan.l1_buffer [full], %1 : !tileplan.l1_buffer [full]) after [#task.dep<@dm0_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %c10_3 = arith.constant 10 : index
            %c9 = arith.constant 9 : index
            %c10_4 = arith.constant 10 : index
            %16 = affine.apply #map2()
            %c0_5 = arith.constant 0 : index
            %c10_6 = arith.constant 10 : index
            %c0_7 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %17 = affine.apply #map2()
            %c0_8 = arith.constant 0 : index
            %c0_9 = arith.constant 0 : index
            %c0_10 = arith.constant 0 : index
            %c0_11 = arith.constant 0 : index
            %18 = d2mtile.fill "2.000000e-01" -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %19 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %20 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %21 = tileflow.read_l1_buffer %0 at(%19, %20) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %22 = affine.apply #map3(%arg5)
                %23 = tileflow.read_l1_buffer %6 at(%22, %16) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %24 = arith.cmpi sge, %arg5, %c0 : index
                %25 = arith.cmpi slt, %arg5, %c10_3 : index
                %26 = arith.andi %24, %25 : i1
                %27 = arith.cmpi sge, %arg6, %c9 : index
                %28 = arith.cmpi slt, %arg6, %c10_4 : index
                %29 = arith.andi %27, %28 : i1
                %30 = arith.andi %26, %29 : i1
                %31 = scf.if %30 -> (!d2mtile.tile<f32>) {
                  scf.yield %23 : !d2mtile.tile<f32>
                } else {
                  %84 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %85 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                  %86 = tileflow.read_l1_buffer %0 at(%84, %85) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %86 : !d2mtile.tile<f32>
                } {task.engine_label = "compute"}
                %32 = d2mtile.unary tile_transpose %21 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) to %32 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %33 = d2mtile.unary tile_transpose %31 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) to %33 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %34 = affine.apply #map3(%arg5)
                %35 = tileflow.read_l1_buffer %7 at(%34, %17) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %36 = arith.cmpi sge, %arg5, %c0_5 : index
                %37 = arith.cmpi slt, %arg5, %c10_6 : index
                %38 = arith.andi %36, %37 : i1
                %39 = arith.cmpi sge, %arg6, %c0_7 : index
                %40 = arith.cmpi slt, %arg6, %c1 : index
                %41 = arith.andi %39, %40 : i1
                %42 = arith.andi %38, %41 : i1
                %43 = scf.if %42 -> (!d2mtile.tile<f32>) {
                  scf.yield %35 : !d2mtile.tile<f32>
                } else {
                  %84 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %85 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                  %86 = tileflow.read_l1_buffer %0 at(%84, %85) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %86 : !d2mtile.tile<f32>
                } {task.engine_label = "compute"}
                %44 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %45 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %46 = tileflow.read_l1_buffer %0 at(%44, %45) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %47 = d2mtile.unary tile_transpose %46 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) to %47 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %48 = d2mtile.unary tile_transpose %43 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) to %48 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %49 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %50 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %51 = tileflow.read_l1_buffer %0 at(%49, %50) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %52 = tileflow.read_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %53 = tileflow.read_l1_buffer %15 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %54 = d2mtile.binary tile_add %53, %52 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %55 = d2mtile.unary tile_transpose %54 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %56 = tileflow.read_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %57 = d2mtile.binary tile_add %51, %56 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %58 = tileflow.read_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %59 = d2mtile.binary tile_add %57, %58 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %60 = d2mtile.binary tile_add %59, %55 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %61 = d2mtile.binary tile_mul %60, %18 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %c9_12 = arith.constant 9 : index
                %62 = arith.cmpi eq, %arg5, %c9_12 : index
                %c1_13 = arith.constant 1 : index
                %c0_14 = arith.constant 0 : index
                %63 = arith.select %62, %c1_13, %c0_14 : index
                %c0_15 = arith.constant 0 : index
                %64 = tileflow.read_l1_buffer %2 at(%63, %c0_15) : !tileplan.l1_buffer -> !tileflow.mask<f32>
                %65 = affine.apply #map(%arg1, %arg5)
                %66 = affine.apply #map10()[%65]
                %67 = arith.cmpi slt, %66, %c0_8 : index
                %68 = affine.apply #map1(%arg1, %arg5)
                %69 = affine.apply #map11()[%68]
                %70 = arith.cmpi slt, %69, %c0_9 : index
                %71 = arith.ori %67, %70 : i1
                %72 = tileflow.compute_select %61 with %51 active(%64 : !tileflow.mask<f32>) if(%71) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
                %c0_16 = arith.constant 0 : index
                %c9_17 = arith.constant 9 : index
                %73 = arith.cmpi eq, %arg6, %c9_17 : index
                %c1_18 = arith.constant 1 : index
                %c0_19 = arith.constant 0 : index
                %74 = arith.select %73, %c1_18, %c0_19 : index
                %75 = tileflow.read_l1_buffer %3 at(%c0_16, %74) : !tileplan.l1_buffer -> !tileflow.mask<f32>
                %76 = affine.apply #map(%arg2, %arg6)
                %77 = affine.apply #map10()[%76]
                %78 = arith.cmpi slt, %77, %c0_10 : index
                %79 = affine.apply #map1(%arg2, %arg6)
                %80 = affine.apply #map11()[%79]
                %81 = arith.cmpi slt, %80, %c0_11 : index
                %82 = arith.ori %78, %81 : i1
                %83 = tileflow.compute_select %72 with %51 active(%75 : !tileflow.mask<f32>) if(%82) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %1 at(%arg5, %arg6) epoch(%arg4) to %83 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
          }
          task.group @dm0_3 engine(<dm0>) domain(<phase>) reads(%1 : !tileplan.l1_buffer [full]) writes(%0 : !tileplan.l1_buffer [full]) after [#task.dep<@compute_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %16 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %17 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %18 = tileflow.read_l1_buffer %1 at(%16, %17) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %18 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

