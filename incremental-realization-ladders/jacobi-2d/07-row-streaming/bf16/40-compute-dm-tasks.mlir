#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<(d0) -> (d0 mod 10)>
#map4 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 - 1) mod 10)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 mod 10)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10)>
#map7 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 + 1) mod 10)>
#map8 = affine_map<(d0, d1, d2, d3) -> ((d3 + d1 * 10 + 1) mod 10)>
#map9 = affine_map<(d0, d1, d2, d3) -> ((d3 + d1 * 10 - 1) mod 10)>
#map10 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * 32 - 1)>
#map11 = affine_map<()[s0] -> (s0)>
#map12 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * -32 + 3198)>
#map13 = affine_map<()[s0] -> (s0 - 31)>
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
        %3 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(1) {frame_axis = #tileplan.axis<col>} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(2) {frame_axis = #tileplan.axis<col>} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(3) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(4) {frame_axis = #tileplan.axis<row>} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(5) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(6) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(7) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(8) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        %11 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(9) {frame_axis = #tileplan.axis<row_col>} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<3200x3200xbf16>) writes(%arg3 : memref<3200x3200xbf16>) {producer = "dm"}
        tileplan.bind_persistent %1 : !tileplan.l1_buffer reads() writes() {producer = "compute"}
        tileplan.temporal_for (%arg4) in (0, 1000, 1) {
          task.group @compute_0 engine(<compute>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes(%1 : !tileplan.l1_buffer [full]) {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %12 = affine.apply #map(%arg5, %arg6)
                %13 = affine.apply #map1(%arg5, %arg6)
                %14 = tileflow.read_l1_buffer %0 at(%12, %13) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %15 = d2mtile.unary tile_transpose %14 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %1 at(%arg5, %arg6) epoch(%arg4) to %15 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes(%3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full]) {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %true = arith.constant true
            %true_3 = arith.constant true
            %12 = arith.andi %true, %true_3 : i1
            %true_4 = arith.constant true
            %true_5 = arith.constant true
            %13 = arith.andi %true_4, %true_5 : i1
            %c-1 = arith.constant -1 : index
            %14 = arith.addi %arg1, %c-1 : index
            %c0 = arith.constant 0 : index
            %15 = arith.maxsi %14, %c0 : index
            %c9 = arith.constant 9 : index
            %c0_6 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %16 = arith.addi %arg1, %c1 : index
            %c9_7 = arith.constant 9 : index
            %17 = arith.minsi %16, %c9_7 : index
            %c0_8 = arith.constant 0 : index
            %c0_9 = arith.constant 0 : index
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            %c0_12 = arith.constant 0 : index
            %c10_13 = arith.constant 10 : index
            %18 = affine.apply #map2()
            %c9_14 = arith.constant 9 : index
            %c10_15 = arith.constant 10 : index
            %c0_16 = arith.constant 0 : index
            %c10_17 = arith.constant 10 : index
            %19 = affine.apply #map2()
            %true_18 = arith.constant true
            %true_19 = arith.constant true
            tileflow.bulk_transfer %0 into %3 source_core(%15, %arg2) source_slice(%c9, %c0_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %4 source_core(%17, %arg2) source_slice(%c0_8, %c0_9) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %20 = arith.cmpi sge, %arg5, %c0_10 : index
                %21 = arith.cmpi slt, %arg5, %c1_11 : index
                %22 = arith.andi %20, %21 : i1
                %23 = arith.cmpi sge, %arg6, %c0_12 : index
                %24 = arith.cmpi slt, %arg6, %c10_13 : index
                %25 = arith.andi %23, %24 : i1
                %26 = arith.andi %22, %25 : i1
                %27 = affine.apply #map3(%arg6)
                %28 = tileflow.read_l1_buffer %3 at(%18, %27) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %29 = scf.if %26 -> (!d2mtile.tile<bf16>) {
                  scf.yield %28 : !d2mtile.tile<bf16>
                } else {
                  %52 = affine.apply #map4(%arg1, %arg2, %arg5, %arg6)
                  %53 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %54 = tileflow.read_l1_buffer %0 at(%52, %53) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %54 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %30 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %31 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %32 = tileflow.read_l1_buffer %0 at(%30, %31) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %33 = tileflow.roll %32 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %34 = tileflow.lane_copy %33 with %32 if(%true_18) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %35 = tileflow.lane_copy %34 with %29 if(%12) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %7 at(%arg5, %arg6) epoch(%arg4) to %35 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                %36 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %37 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %38 = tileflow.read_l1_buffer %0 at(%36, %37) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %39 = arith.cmpi sge, %arg5, %c9_14 : index
                %40 = arith.cmpi slt, %arg5, %c10_15 : index
                %41 = arith.andi %39, %40 : i1
                %42 = arith.cmpi sge, %arg6, %c0_16 : index
                %43 = arith.cmpi slt, %arg6, %c10_17 : index
                %44 = arith.andi %42, %43 : i1
                %45 = arith.andi %41, %44 : i1
                %46 = affine.apply #map3(%arg6)
                %47 = tileflow.read_l1_buffer %4 at(%19, %46) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %48 = scf.if %45 -> (!d2mtile.tile<bf16>) {
                  scf.yield %47 : !d2mtile.tile<bf16>
                } else {
                  %52 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                  %53 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %54 = tileflow.read_l1_buffer %0 at(%52, %53) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %54 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %49 = tileflow.roll %38 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %50 = tileflow.lane_copy %49 with %38 if(%true_19) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %51 = tileflow.lane_copy %50 with %48 if(%13) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) to %51 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
          task.group @dm1_0 engine(<dm1>) domain(<phase>) reads(%1 : !tileplan.l1_buffer [full]) writes(%5 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %9 : !tileplan.l1_buffer [full], %10 : !tileplan.l1_buffer [full]) after [#task.dep<@compute_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c1 = arith.constant 1 : index
            %12 = arith.addi %arg2, %c1 : index
            %c9 = arith.constant 9 : index
            %13 = arith.minsi %12, %c9 : index
            %c0 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %c-1 = arith.constant -1 : index
            %14 = arith.addi %arg2, %c-1 : index
            %c0_4 = arith.constant 0 : index
            %15 = arith.maxsi %14, %c0_4 : index
            %c0_5 = arith.constant 0 : index
            %c9_6 = arith.constant 9 : index
            %true = arith.constant true
            %true_7 = arith.constant true
            %16 = arith.andi %true, %true_7 : i1
            %c0_8 = arith.constant 0 : index
            %c10_9 = arith.constant 10 : index
            %c9_10 = arith.constant 9 : index
            %c10_11 = arith.constant 10 : index
            %17 = affine.apply #map2()
            %true_12 = arith.constant true
            %true_13 = arith.constant true
            %18 = arith.andi %true_12, %true_13 : i1
            %c0_14 = arith.constant 0 : index
            %c10_15 = arith.constant 10 : index
            %c0_16 = arith.constant 0 : index
            %c1_17 = arith.constant 1 : index
            %19 = affine.apply #map2()
            %true_18 = arith.constant true
            %true_19 = arith.constant true
            tileflow.bulk_transfer %1 into %5 source_core(%arg1, %13) source_slice(%c0, %c0_3) epoch(%arg4) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %1 into %6 source_core(%arg1, %15) source_slice(%c0_5, %c9_6) epoch(%arg4) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %20 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %21 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %22 = tileflow.read_l1_buffer %1 at(%20, %21) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %23 = tileflow.roll %22 axis row by 1 {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %24 = tileflow.lane_copy %23 with %22 if(%true_18) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %25 = arith.cmpi sge, %arg5, %c0_8 : index
                %26 = arith.cmpi slt, %arg5, %c10_9 : index
                %27 = arith.andi %25, %26 : i1
                %28 = arith.cmpi sge, %arg6, %c9_10 : index
                %29 = arith.cmpi slt, %arg6, %c10_11 : index
                %30 = arith.andi %28, %29 : i1
                %31 = arith.andi %27, %30 : i1
                %32 = affine.apply #map3(%arg5)
                %33 = tileflow.read_l1_buffer %5 at(%32, %17) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %34 = scf.if %31 -> (!d2mtile.tile<bf16>) {
                  scf.yield %33 : !d2mtile.tile<bf16>
                } else {
                  %52 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %53 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                  %54 = tileflow.read_l1_buffer %1 at(%52, %53) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %54 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %35 = tileflow.lane_copy %24 with %34 if(%16) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) to %35 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                %36 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %37 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %38 = tileflow.read_l1_buffer %1 at(%36, %37) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %39 = tileflow.roll %38 axis row by -1 {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %40 = tileflow.lane_copy %39 with %38 if(%true_19) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %41 = arith.cmpi sge, %arg5, %c0_14 : index
                %42 = arith.cmpi slt, %arg5, %c10_15 : index
                %43 = arith.andi %41, %42 : i1
                %44 = arith.cmpi sge, %arg6, %c0_16 : index
                %45 = arith.cmpi slt, %arg6, %c1_17 : index
                %46 = arith.andi %44, %45 : i1
                %47 = arith.andi %43, %46 : i1
                %48 = affine.apply #map3(%arg5)
                %49 = tileflow.read_l1_buffer %6 at(%48, %19) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %50 = scf.if %47 -> (!d2mtile.tile<bf16>) {
                  scf.yield %49 : !d2mtile.tile<bf16>
                } else {
                  %52 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %53 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                  %54 = tileflow.read_l1_buffer %1 at(%52, %53) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %54 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %51 = tileflow.lane_copy %40 with %50 if(%18) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) to %51 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
          task.group @compute_0_1 engine(<compute>) domain(<phase>) reads(%3 : !tileplan.l1_buffer [full], %0 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %1 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %9 : !tileplan.l1_buffer [full], %10 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full]) writes(%2 : !tileplan.l1_buffer [full]) {
            %12 = d2mtile.fill "2.000000e-01" -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %13 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %14 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %15 = tileflow.read_l1_buffer %0 at(%13, %14) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %16 = tileflow.read_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %17 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %18 = d2mtile.binary tile_add %17, %16 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %19 = d2mtile.unary tile_transpose %18 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %20 = tileflow.read_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %21 = d2mtile.binary tile_add %15, %20 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %22 = tileflow.read_l1_buffer %7 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %23 = d2mtile.binary tile_add %21, %22 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %24 = d2mtile.binary tile_add %23, %19 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                %25 = d2mtile.binary tile_mul %24, %12 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %2 at(%arg5, %arg6) epoch(%arg4) to %25 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
          }
          task.group @dm0_1 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full], %2 : !tileplan.l1_buffer [full], %11 : !tileplan.l1_buffer [full]) writes(%11 : !tileplan.l1_buffer [full], %0 : !tileplan.l1_buffer [full]) after [#task.dep<@compute_0_1, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %true = arith.constant true
            %c0_3 = arith.constant 0 : index
            %true_4 = arith.constant true
            %c0_5 = arith.constant 0 : index
            %true_6 = arith.constant true
            %c0_7 = arith.constant 0 : index
            %true_8 = arith.constant true
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %12 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %13 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %14 = tileflow.read_l1_buffer %0 at(%12, %13) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %15 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %16 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %17 = tileflow.read_l1_buffer %2 at(%15, %16) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %18 = affine.apply #map10(%arg1, %arg5)
                %19 = affine.apply #map11()[%18]
                %20 = arith.cmpi sge, %19, %c0 : index
                %21 = arith.xori %20, %true : i1
                %22 = affine.apply #map12(%arg1, %arg5)
                %23 = affine.apply #map13()[%22]
                %24 = arith.cmpi sge, %23, %c0_3 : index
                %25 = arith.xori %24, %true_4 : i1
                %26 = arith.ori %21, %25 : i1
                %27 = affine.apply #map10(%arg2, %arg6)
                %28 = affine.apply #map11()[%27]
                %29 = arith.cmpi sge, %28, %c0_5 : index
                %30 = arith.xori %29, %true_6 : i1
                %31 = arith.ori %26, %30 : i1
                %32 = affine.apply #map12(%arg2, %arg6)
                %33 = affine.apply #map13()[%32]
                %34 = arith.cmpi sge, %33, %c0_7 : index
                %35 = arith.xori %34, %true_8 : i1
                %36 = arith.ori %31, %35 : i1
                scf.if %36 {
                  %37 = tileflow.lane_copy %17 with %14 if(%21) axis row src_lane 0 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                  %38 = tileflow.lane_copy %37 with %14 if(%25) axis row src_lane 31 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                  %39 = tileflow.lane_copy %38 with %14 if(%30) axis col src_lane 0 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                  %40 = tileflow.lane_copy %39 with %14 if(%35) axis col src_lane 31 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                  tileflow.write_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) to %40 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                  %41 = tileflow.read_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %41 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                } else {
                  tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %17 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
              }
            }
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

