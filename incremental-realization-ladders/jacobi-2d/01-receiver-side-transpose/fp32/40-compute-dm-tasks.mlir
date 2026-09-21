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
        %16 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(15) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %17 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(16) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<3200x3200xf32>) writes(%arg3 : memref<3200x3200xf32>) {producer = "dm"}
        task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes(%2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) {
          %c1 = arith.constant 1 : index
          %c1_1 = arith.constant 1 : index
          tileplan.shard_for (%arg4) in (%c1) axis <row> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %18 = affine.apply #map(%arg1, %c0_3)
            %19 = tileflow.row_mask symbols(%18) set #set invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %2 at(%c0, %c0_2) to %19 : !tileplan.l1_buffer to !tileflow.mask<f32>
            %c1_4 = arith.constant 1 : index
            %c0_5 = arith.constant 0 : index
            %c9 = arith.constant 9 : index
            %20 = affine.apply #map1(%arg1, %c9)
            %21 = tileflow.row_mask symbols(%20) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %2 at(%c1_4, %c0_5) to %21 : !tileplan.l1_buffer to !tileflow.mask<f32>
          }
          tileplan.shard_for (%arg4) in (%c1_1) axis <col> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %18 = affine.apply #map(%arg2, %c0_3)
            %19 = tileflow.col_mask symbols(%18) set #set invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %3 at(%c0, %c0_2) to %19 : !tileplan.l1_buffer to !tileflow.mask<f32>
            %c0_4 = arith.constant 0 : index
            %c1_5 = arith.constant 1 : index
            %c9 = arith.constant 9 : index
            %20 = affine.apply #map1(%arg2, %c9)
            %21 = tileflow.col_mask symbols(%20) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %3 at(%c0_4, %c1_5) to %21 : !tileplan.l1_buffer to !tileflow.mask<f32>
          }
        }
        tileplan.temporal_for (%arg4) in (0, 1000, 1) {
          task.group @dm0_0_1 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full], %10 : !tileplan.l1_buffer [full], %12 : !tileplan.l1_buffer [full], %14 : !tileplan.l1_buffer [full], %16 : !tileplan.l1_buffer [full]) writes(%4 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %9 : !tileplan.l1_buffer [full], %11 : !tileplan.l1_buffer [full], %13 : !tileplan.l1_buffer [full], %15 : !tileplan.l1_buffer [full], %17 : !tileplan.l1_buffer [full]) {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %true = arith.constant true
            %true_3 = arith.constant true
            %18 = arith.andi %true, %true_3 : i1
            %true_4 = arith.constant true
            %true_5 = arith.constant true
            %19 = arith.andi %true_4, %true_5 : i1
            %true_6 = arith.constant true
            %true_7 = arith.constant true
            %20 = arith.andi %true_6, %true_7 : i1
            %true_8 = arith.constant true
            %true_9 = arith.constant true
            %21 = arith.andi %true_8, %true_9 : i1
            %c-1 = arith.constant -1 : index
            %22 = arith.addi %arg1, %c-1 : index
            %c0 = arith.constant 0 : index
            %23 = arith.maxsi %22, %c0 : index
            %c9 = arith.constant 9 : index
            %c0_10 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %24 = arith.addi %arg1, %c1 : index
            %c9_11 = arith.constant 9 : index
            %25 = arith.minsi %24, %c9_11 : index
            %c0_12 = arith.constant 0 : index
            %c0_13 = arith.constant 0 : index
            %c1_14 = arith.constant 1 : index
            %26 = arith.addi %arg2, %c1_14 : index
            %c9_15 = arith.constant 9 : index
            %27 = arith.minsi %26, %c9_15 : index
            %c0_16 = arith.constant 0 : index
            %c0_17 = arith.constant 0 : index
            %c-1_18 = arith.constant -1 : index
            %28 = arith.addi %arg2, %c-1_18 : index
            %c0_19 = arith.constant 0 : index
            %29 = arith.maxsi %28, %c0_19 : index
            %c0_20 = arith.constant 0 : index
            %c9_21 = arith.constant 9 : index
            %c0_22 = arith.constant 0 : index
            %c1_23 = arith.constant 1 : index
            %c0_24 = arith.constant 0 : index
            %c10_25 = arith.constant 10 : index
            %30 = affine.apply #map2()
            %c9_26 = arith.constant 9 : index
            %c10_27 = arith.constant 10 : index
            %c0_28 = arith.constant 0 : index
            %c10_29 = arith.constant 10 : index
            %31 = affine.apply #map2()
            %c0_30 = arith.constant 0 : index
            %c10_31 = arith.constant 10 : index
            %c9_32 = arith.constant 9 : index
            %c10_33 = arith.constant 10 : index
            %32 = affine.apply #map2()
            %c0_34 = arith.constant 0 : index
            %c10_35 = arith.constant 10 : index
            %c0_36 = arith.constant 0 : index
            %c1_37 = arith.constant 1 : index
            %33 = affine.apply #map2()
            %true_38 = arith.constant true
            %true_39 = arith.constant true
            %true_40 = arith.constant true
            %true_41 = arith.constant true
            tileflow.bulk_transfer %0 into %4 source_core(%23, %arg2) source_slice(%c9, %c0_10) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %5 source_core(%25, %arg2) source_slice(%c0_12, %c0_13) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %6 source_core(%arg1, %27) source_slice(%c0_16, %c0_17) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %7 source_core(%arg1, %29) source_slice(%c0_20, %c9_21) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %34 = arith.cmpi sge, %arg5, %c0_22 : index
                %35 = arith.cmpi slt, %arg5, %c1_23 : index
                %36 = arith.andi %34, %35 : i1
                %37 = arith.cmpi sge, %arg6, %c0_24 : index
                %38 = arith.cmpi slt, %arg6, %c10_25 : index
                %39 = arith.andi %37, %38 : i1
                %40 = arith.andi %36, %39 : i1
                %41 = affine.apply #map3(%arg6)
                %42 = tileflow.read_l1_buffer %4 at(%30, %41) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %43 = scf.if %40 -> (!d2mtile.tile<f32>) {
                  scf.yield %42 : !d2mtile.tile<f32>
                } else {
                  %98 = affine.apply #map4(%arg1, %arg2, %arg5, %arg6)
                  %99 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %100 = tileflow.read_l1_buffer %0 at(%98, %99) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %100 : !d2mtile.tile<f32>
                } {task.engine_label = "dm"}
                %44 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %45 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %46 = tileflow.read_l1_buffer %0 at(%44, %45) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %47 = tileflow.roll %46 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %48 = tileflow.lane_copy %47 with %46 if(%true_38) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                %49 = tileflow.lane_copy %48 with %43 if(%18) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) to %49 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %50 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %51 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %52 = tileflow.read_l1_buffer %0 at(%50, %51) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %53 = arith.cmpi sge, %arg5, %c9_26 : index
                %54 = arith.cmpi slt, %arg5, %c10_27 : index
                %55 = arith.andi %53, %54 : i1
                %56 = arith.cmpi sge, %arg6, %c0_28 : index
                %57 = arith.cmpi slt, %arg6, %c10_29 : index
                %58 = arith.andi %56, %57 : i1
                %59 = arith.andi %55, %58 : i1
                %60 = affine.apply #map3(%arg6)
                %61 = tileflow.read_l1_buffer %5 at(%31, %60) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %62 = scf.if %59 -> (!d2mtile.tile<f32>) {
                  scf.yield %61 : !d2mtile.tile<f32>
                } else {
                  %98 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                  %99 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %100 = tileflow.read_l1_buffer %0 at(%98, %99) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %100 : !d2mtile.tile<f32>
                } {task.engine_label = "dm"}
                %63 = tileflow.roll %52 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %64 = tileflow.lane_copy %63 with %52 if(%true_39) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                %65 = tileflow.lane_copy %64 with %62 if(%19) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) to %65 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %66 = arith.cmpi sge, %arg5, %c0_30 : index
                %67 = arith.cmpi slt, %arg5, %c10_31 : index
                %68 = arith.andi %66, %67 : i1
                %69 = arith.cmpi sge, %arg6, %c9_32 : index
                %70 = arith.cmpi slt, %arg6, %c10_33 : index
                %71 = arith.andi %69, %70 : i1
                %72 = arith.andi %68, %71 : i1
                %73 = affine.apply #map3(%arg5)
                %74 = tileflow.read_l1_buffer %6 at(%73, %32) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %75 = scf.if %72 -> (!d2mtile.tile<f32>) {
                  scf.yield %74 : !d2mtile.tile<f32>
                } else {
                  %98 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %99 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                  %100 = tileflow.read_l1_buffer %0 at(%98, %99) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %100 : !d2mtile.tile<f32>
                } {task.engine_label = "dm"}
                %76 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %77 = tileflow.roll %76 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %78 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %79 = tileflow.lane_copy %77 with %78 if(%true_40) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) to %79 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %80 = tileflow.read_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %81 = tileflow.lane_copy %80 with %75 if(%20) axis col src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) to %81 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %82 = arith.cmpi sge, %arg5, %c0_34 : index
                %83 = arith.cmpi slt, %arg5, %c10_35 : index
                %84 = arith.andi %82, %83 : i1
                %85 = arith.cmpi sge, %arg6, %c0_36 : index
                %86 = arith.cmpi slt, %arg6, %c1_37 : index
                %87 = arith.andi %85, %86 : i1
                %88 = arith.andi %84, %87 : i1
                %89 = affine.apply #map3(%arg5)
                %90 = tileflow.read_l1_buffer %7 at(%89, %33) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %91 = scf.if %88 -> (!d2mtile.tile<f32>) {
                  scf.yield %90 : !d2mtile.tile<f32>
                } else {
                  %98 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %99 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                  %100 = tileflow.read_l1_buffer %0 at(%98, %99) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %100 : !d2mtile.tile<f32>
                } {task.engine_label = "dm"}
                %92 = tileflow.read_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %93 = tileflow.roll %92 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %94 = tileflow.read_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %95 = tileflow.lane_copy %93 with %94 if(%true_41) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %15 at(%arg5, %arg6) epoch(%arg4) to %95 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %96 = tileflow.read_l1_buffer %16 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %97 = tileflow.lane_copy %96 with %91 if(%21) axis col src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %17 at(%arg5, %arg6) epoch(%arg4) to %97 : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
          }
          task.group @compute_0 engine(<compute>) domain(<phase>) reads(%4 : !tileplan.l1_buffer [full], %0 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %11 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %15 : !tileplan.l1_buffer [full], %17 : !tileplan.l1_buffer [full], %13 : !tileplan.l1_buffer [full], %9 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) writes(%10 : !tileplan.l1_buffer [full], %12 : !tileplan.l1_buffer [full], %14 : !tileplan.l1_buffer [full], %16 : !tileplan.l1_buffer [full], %1 : !tileplan.l1_buffer [full]) after [#task.dep<@dm0_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %c0_5 = arith.constant 0 : index
            %18 = d2mtile.fill "2.000000e-01" -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %19 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %20 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %21 = tileflow.read_l1_buffer %0 at(%19, %20) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %22 = d2mtile.unary tile_transpose %21 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) to %22 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %23 = tileflow.read_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %24 = d2mtile.unary tile_transpose %23 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) to %24 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %25 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %26 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %27 = tileflow.read_l1_buffer %0 at(%25, %26) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %28 = d2mtile.unary tile_transpose %27 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) to %28 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %29 = tileflow.read_l1_buffer %15 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %30 = d2mtile.unary tile_transpose %29 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %16 at(%arg5, %arg6) epoch(%arg4) to %30 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                %31 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %32 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %33 = tileflow.read_l1_buffer %0 at(%31, %32) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %34 = tileflow.read_l1_buffer %17 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %35 = d2mtile.binary tile_add %33, %34 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %36 = tileflow.read_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %37 = d2mtile.binary tile_add %35, %36 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %38 = tileflow.read_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %39 = d2mtile.binary tile_add %37, %38 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %40 = tileflow.read_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %41 = d2mtile.binary tile_add %39, %40 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %42 = d2mtile.binary tile_mul %41, %18 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<f32>
                %c9 = arith.constant 9 : index
                %43 = arith.cmpi eq, %arg5, %c9 : index
                %c1 = arith.constant 1 : index
                %c0_6 = arith.constant 0 : index
                %44 = arith.select %43, %c1, %c0_6 : index
                %c0_7 = arith.constant 0 : index
                %45 = tileflow.read_l1_buffer %2 at(%44, %c0_7) : !tileplan.l1_buffer -> !tileflow.mask<f32>
                %46 = affine.apply #map(%arg1, %arg5)
                %47 = affine.apply #map10()[%46]
                %48 = arith.cmpi slt, %47, %c0 : index
                %49 = affine.apply #map1(%arg1, %arg5)
                %50 = affine.apply #map11()[%49]
                %51 = arith.cmpi slt, %50, %c0_3 : index
                %52 = arith.ori %48, %51 : i1
                %53 = tileflow.compute_select %42 with %33 active(%45 : !tileflow.mask<f32>) if(%52) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
                %c0_8 = arith.constant 0 : index
                %c9_9 = arith.constant 9 : index
                %54 = arith.cmpi eq, %arg6, %c9_9 : index
                %c1_10 = arith.constant 1 : index
                %c0_11 = arith.constant 0 : index
                %55 = arith.select %54, %c1_10, %c0_11 : index
                %56 = tileflow.read_l1_buffer %3 at(%c0_8, %55) : !tileplan.l1_buffer -> !tileflow.mask<f32>
                %57 = affine.apply #map(%arg2, %arg6)
                %58 = affine.apply #map10()[%57]
                %59 = arith.cmpi slt, %58, %c0_4 : index
                %60 = affine.apply #map1(%arg2, %arg6)
                %61 = affine.apply #map11()[%60]
                %62 = arith.cmpi slt, %61, %c0_5 : index
                %63 = arith.ori %59, %62 : i1
                %64 = tileflow.compute_select %53 with %33 active(%56 : !tileflow.mask<f32>) if(%63) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %1 at(%arg5, %arg6) epoch(%arg4) to %64 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
          }
          task.group @dm0_5 engine(<dm0>) domain(<phase>) reads(%1 : !tileplan.l1_buffer [full]) writes(%0 : !tileplan.l1_buffer [full]) after [#task.dep<@compute_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %18 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %19 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %20 = tileflow.read_l1_buffer %1 at(%18, %19) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %20 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

