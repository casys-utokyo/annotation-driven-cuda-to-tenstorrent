#map = affine_map<(d0, d1) -> ((d0 * 10 + d1) * 32 - 1)>
#map1 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * -32 + 3198)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<(d0) -> (d0 mod 10)>
#map4 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 - 1) mod 10)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 mod 10)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10)>
#map7 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 + 1) mod 10)>
#map8 = affine_map<(d0, d1) -> (d0)>
#map9 = affine_map<(d0, d1) -> (d1)>
#map10 = affine_map<(d0, d1, d2, d3) -> ((d3 + d1 * 10 + 1) mod 10)>
#map11 = affine_map<(d0, d1, d2, d3) -> ((d3 + d1 * 10 - 1) mod 10)>
#map12 = affine_map<()[s0] -> (s0)>
#map13 = affine_map<()[s0] -> (s0 - 31)>
#set = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 >= 0)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg1, %arg2) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(10, 10) order(0) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10]} : !tileplan.l1_buffer
      %1 = tileplan.persistent_l1 name "frame.converted" elem bf16 tile(32, 32) domain(10, 10) order(1) : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<3200x3200xbf16>) {
      ^bb0(%arg3: memref<3200x3200xbf16>):
        %2 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(0) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(2, 1) order(1) {frame_axis = #tileplan.axis<row>, task.frame_access = "replay", task.reuse = 1000 : i64} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 2) order(2) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 10 : i64} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(3) {frame_axis = #tileplan.axis<col>} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(4) {frame_axis = #tileplan.axis<col>} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(5) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(6) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(7) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(8) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %11 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(9) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %12 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(10) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<3200x3200xbf16>) writes(%arg3 : memref<3200x3200xbf16>) {producer = "dm"}
        tileplan.bind_persistent %1 : !tileplan.l1_buffer reads() writes() {producer = "compute"}
        task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes(%3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full]) {
          %c1 = arith.constant 1 : index
          %c1_1 = arith.constant 1 : index
          tileplan.shard_for (%arg4) in (%c1) axis <row> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %13 = affine.apply #map(%arg1, %c0_3)
            %14 = tileflow.row_mask symbols(%13) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %3 at(%c0, %c0_2) to %14 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            %c1_4 = arith.constant 1 : index
            %c0_5 = arith.constant 0 : index
            %c9 = arith.constant 9 : index
            %15 = affine.apply #map1(%arg1, %c9)
            %16 = tileflow.row_mask symbols(%15) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %3 at(%c1_4, %c0_5) to %16 : !tileplan.l1_buffer to !tileflow.mask<bf16>
          }
          tileplan.shard_for (%arg4) in (%c1_1) axis <col> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %13 = affine.apply #map(%arg2, %c0_3)
            %14 = tileflow.col_mask symbols(%13) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %4 at(%c0, %c0_2) to %14 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            %c0_4 = arith.constant 0 : index
            %c1_5 = arith.constant 1 : index
            %c9 = arith.constant 9 : index
            %15 = affine.apply #map1(%arg2, %c9)
            %16 = tileflow.col_mask symbols(%15) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %4 at(%c0_4, %c1_5) to %16 : !tileplan.l1_buffer to !tileflow.mask<bf16>
          }
        }
        tileplan.temporal_for (%arg4) in (0, 1000, 1) {
          task.group @dm0_0_1 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes(%5 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %9 : !tileplan.l1_buffer [full], %10 : !tileplan.l1_buffer [full]) {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %true = arith.constant true
            %true_3 = arith.constant true
            %13 = arith.andi %true, %true_3 : i1
            %true_4 = arith.constant true
            %true_5 = arith.constant true
            %14 = arith.andi %true_4, %true_5 : i1
            %c-1 = arith.constant -1 : index
            %15 = arith.addi %arg1, %c-1 : index
            %c0 = arith.constant 0 : index
            %16 = arith.maxsi %15, %c0 : index
            %c9 = arith.constant 9 : index
            %c0_6 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %17 = arith.addi %arg1, %c1 : index
            %c9_7 = arith.constant 9 : index
            %18 = arith.minsi %17, %c9_7 : index
            %c0_8 = arith.constant 0 : index
            %c0_9 = arith.constant 0 : index
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            %c0_12 = arith.constant 0 : index
            %c10_13 = arith.constant 10 : index
            %19 = affine.apply #map2()
            %c9_14 = arith.constant 9 : index
            %c10_15 = arith.constant 10 : index
            %c0_16 = arith.constant 0 : index
            %c10_17 = arith.constant 10 : index
            %20 = affine.apply #map2()
            %true_18 = arith.constant true
            %true_19 = arith.constant true
            tileflow.bulk_transfer %0 into %5 source_core(%16, %arg2) source_slice(%c9, %c0_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %6 source_core(%18, %arg2) source_slice(%c0_8, %c0_9) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %21 = arith.cmpi sge, %arg5, %c0_10 : index
                %22 = arith.cmpi slt, %arg5, %c1_11 : index
                %23 = arith.andi %21, %22 : i1
                %24 = arith.cmpi sge, %arg6, %c0_12 : index
                %25 = arith.cmpi slt, %arg6, %c10_13 : index
                %26 = arith.andi %24, %25 : i1
                %27 = arith.andi %23, %26 : i1
                %28 = affine.apply #map3(%arg6)
                %29 = tileflow.read_l1_buffer %5 at(%19, %28) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %30 = scf.if %27 -> (!d2mtile.tile<bf16>) {
                  scf.yield %29 : !d2mtile.tile<bf16>
                } else {
                  %53 = affine.apply #map4(%arg1, %arg2, %arg5, %arg6)
                  %54 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %55 = tileflow.read_l1_buffer %0 at(%53, %54) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %55 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %31 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %32 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %33 = tileflow.read_l1_buffer %0 at(%31, %32) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %34 = tileflow.roll %33 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %35 = tileflow.lane_copy %34 with %33 if(%true_18) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %36 = tileflow.lane_copy %35 with %30 if(%13) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) to %36 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                %37 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %38 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %39 = tileflow.read_l1_buffer %0 at(%37, %38) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %40 = arith.cmpi sge, %arg5, %c9_14 : index
                %41 = arith.cmpi slt, %arg5, %c10_15 : index
                %42 = arith.andi %40, %41 : i1
                %43 = arith.cmpi sge, %arg6, %c0_16 : index
                %44 = arith.cmpi slt, %arg6, %c10_17 : index
                %45 = arith.andi %43, %44 : i1
                %46 = arith.andi %42, %45 : i1
                %47 = affine.apply #map3(%arg6)
                %48 = tileflow.read_l1_buffer %6 at(%20, %47) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %49 = scf.if %46 -> (!d2mtile.tile<bf16>) {
                  scf.yield %48 : !d2mtile.tile<bf16>
                } else {
                  %53 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                  %54 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %55 = tileflow.read_l1_buffer %0 at(%53, %54) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %55 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %50 = tileflow.roll %39 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %51 = tileflow.lane_copy %50 with %39 if(%true_19) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %52 = tileflow.lane_copy %51 with %49 if(%14) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) to %52 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
          task.group @compute_0 engine(<compute>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes(%1 : !tileplan.l1_buffer [full]) {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %13 = affine.apply #map8(%arg5, %arg6)
                %14 = affine.apply #map9(%arg5, %arg6)
                %15 = tileflow.read_l1_buffer %0 at(%13, %14) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %16 = d2mtile.unary tile_transpose %15 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %1 at(%arg5, %arg6) epoch(%arg4) to %16 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
          task.group @dm1_0 engine(<dm1>) domain(<phase>) reads(%1 : !tileplan.l1_buffer [full]) writes(%7 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %11 : !tileplan.l1_buffer [full], %12 : !tileplan.l1_buffer [full]) after [#task.dep<@compute_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c1 = arith.constant 1 : index
            %13 = arith.addi %arg2, %c1 : index
            %c9 = arith.constant 9 : index
            %14 = arith.minsi %13, %c9 : index
            %c0 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %c-1 = arith.constant -1 : index
            %15 = arith.addi %arg2, %c-1 : index
            %c0_4 = arith.constant 0 : index
            %16 = arith.maxsi %15, %c0_4 : index
            %c0_5 = arith.constant 0 : index
            %c9_6 = arith.constant 9 : index
            %true = arith.constant true
            %true_7 = arith.constant true
            %17 = arith.andi %true, %true_7 : i1
            %c0_8 = arith.constant 0 : index
            %c10_9 = arith.constant 10 : index
            %c9_10 = arith.constant 9 : index
            %c10_11 = arith.constant 10 : index
            %18 = affine.apply #map2()
            %true_12 = arith.constant true
            %true_13 = arith.constant true
            %19 = arith.andi %true_12, %true_13 : i1
            %c0_14 = arith.constant 0 : index
            %c10_15 = arith.constant 10 : index
            %c0_16 = arith.constant 0 : index
            %c1_17 = arith.constant 1 : index
            %20 = affine.apply #map2()
            %true_18 = arith.constant true
            %true_19 = arith.constant true
            tileflow.bulk_transfer %1 into %7 source_core(%arg1, %14) source_slice(%c0, %c0_3) epoch(%arg4) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %1 into %8 source_core(%arg1, %16) source_slice(%c0_5, %c9_6) epoch(%arg4) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %21 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %22 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %23 = tileflow.read_l1_buffer %1 at(%21, %22) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %24 = tileflow.roll %23 axis row by 1 {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %25 = tileflow.lane_copy %24 with %23 if(%true_18) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %26 = arith.cmpi sge, %arg5, %c0_8 : index
                %27 = arith.cmpi slt, %arg5, %c10_9 : index
                %28 = arith.andi %26, %27 : i1
                %29 = arith.cmpi sge, %arg6, %c9_10 : index
                %30 = arith.cmpi slt, %arg6, %c10_11 : index
                %31 = arith.andi %29, %30 : i1
                %32 = arith.andi %28, %31 : i1
                %33 = affine.apply #map3(%arg5)
                %34 = tileflow.read_l1_buffer %7 at(%33, %18) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %35 = scf.if %32 -> (!d2mtile.tile<bf16>) {
                  scf.yield %34 : !d2mtile.tile<bf16>
                } else {
                  %53 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %54 = affine.apply #map10(%arg1, %arg2, %arg5, %arg6)
                  %55 = tileflow.read_l1_buffer %1 at(%53, %54) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %55 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %36 = tileflow.lane_copy %25 with %35 if(%17) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) to %36 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                %37 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %38 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %39 = tileflow.read_l1_buffer %1 at(%37, %38) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %40 = tileflow.roll %39 axis row by -1 {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %41 = tileflow.lane_copy %40 with %39 if(%true_19) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %42 = arith.cmpi sge, %arg5, %c0_14 : index
                %43 = arith.cmpi slt, %arg5, %c10_15 : index
                %44 = arith.andi %42, %43 : i1
                %45 = arith.cmpi sge, %arg6, %c0_16 : index
                %46 = arith.cmpi slt, %arg6, %c1_17 : index
                %47 = arith.andi %45, %46 : i1
                %48 = arith.andi %44, %47 : i1
                %49 = affine.apply #map3(%arg5)
                %50 = tileflow.read_l1_buffer %8 at(%49, %20) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %51 = scf.if %48 -> (!d2mtile.tile<bf16>) {
                  scf.yield %50 : !d2mtile.tile<bf16>
                } else {
                  %53 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %54 = affine.apply #map11(%arg1, %arg2, %arg5, %arg6)
                  %55 = tileflow.read_l1_buffer %1 at(%53, %54) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %55 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %52 = tileflow.lane_copy %41 with %51 if(%19) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) to %52 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
          task.group @compute_0_1 engine(<compute>) domain(<phase>) reads(%5 : !tileplan.l1_buffer [full], %0 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %1 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %11 : !tileplan.l1_buffer [full], %12 : !tileplan.l1_buffer [full], %10 : !tileplan.l1_buffer [full], %9 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full]) writes(%2 : !tileplan.l1_buffer [full]) after [#task.dep<@dm0_0, raw>] {
            %13 = d2mtile.fill "2.000000e-01" -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %c0_5 = arith.constant 0 : index
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %14 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %15 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %16 = tileflow.read_l1_buffer %0 at(%14, %15) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %17 = tileflow.read_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %18 = tileflow.read_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %19 = d2mtile.binary tile_add %18, %17 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %20 = d2mtile.unary tile_transpose %19 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %21 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %22 = d2mtile.binary tile_add %16, %21 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %23 = tileflow.read_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %24 = d2mtile.binary tile_add %22, %23 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %25 = d2mtile.binary tile_add %24, %20 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %26 = d2mtile.binary tile_mul %25, %13 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %c9 = arith.constant 9 : index
                %27 = arith.cmpi eq, %arg5, %c9 : index
                %c1 = arith.constant 1 : index
                %c0_6 = arith.constant 0 : index
                %28 = arith.select %27, %c1, %c0_6 : index
                %c0_7 = arith.constant 0 : index
                %29 = tileflow.read_l1_buffer %3 at(%28, %c0_7) : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %30 = affine.apply #map(%arg1, %arg5)
                %31 = affine.apply #map12()[%30]
                %32 = arith.cmpi slt, %31, %c0 : index
                %33 = affine.apply #map1(%arg1, %arg5)
                %34 = affine.apply #map13()[%33]
                %35 = arith.cmpi slt, %34, %c0_3 : index
                %36 = arith.ori %32, %35 : i1
                %37 = tileflow.compute_select %26 with %16 active(%29 : !tileflow.mask<bf16>) if(%36) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<bf16>
                %c0_8 = arith.constant 0 : index
                %c9_9 = arith.constant 9 : index
                %38 = arith.cmpi eq, %arg6, %c9_9 : index
                %c1_10 = arith.constant 1 : index
                %c0_11 = arith.constant 0 : index
                %39 = arith.select %38, %c1_10, %c0_11 : index
                %40 = tileflow.read_l1_buffer %4 at(%c0_8, %39) : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %41 = affine.apply #map(%arg2, %arg6)
                %42 = affine.apply #map12()[%41]
                %43 = arith.cmpi slt, %42, %c0_4 : index
                %44 = affine.apply #map1(%arg2, %arg6)
                %45 = affine.apply #map13()[%44]
                %46 = arith.cmpi slt, %45, %c0_5 : index
                %47 = arith.ori %43, %46 : i1
                %48 = tileflow.compute_select %37 with %16 active(%40 : !tileflow.mask<bf16>) if(%47) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %2 at(%arg5, %arg6) epoch(%arg4) to %48 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
          task.group @dm0_2 engine(<dm0>) domain(<phase>) reads(%2 : !tileplan.l1_buffer [full]) writes(%0 : !tileplan.l1_buffer [full]) after [#task.dep<@compute_0_1, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %13 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %14 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %15 = tileflow.read_l1_buffer %2 at(%13, %14) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %15 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

