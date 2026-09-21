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
        %2 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(0) {frame_axis = #tileplan.axis<row_col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(1) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(2) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(3) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(4) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 10) order(5) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 10) order(6) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 10) order(7) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 10) order(8) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<row>} : !tileplan.l1_buffer
        %11 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(9) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>} : !tileplan.l1_buffer
        %12 = tileplan.l1_buffer role<intermediate> name "dst_spill_10" elem bf16 tile(32, 32) domain(1, 1) order(10) {tt.dst_spill} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<3200x3200xbf16>) writes(%arg3 : memref<3200x3200xbf16>) {producer = "dm"}
        tileplan.bind_persistent %1 : !tileplan.l1_buffer reads() writes() {producer = "compute"}
        tileplan.temporal_for (%arg4) in (0, 1000, 1) {
          task.group @compute_0 engine(<compute>) domain(<phase>) reads() writes() {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %1 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %13 = affine.apply #map(%arg5, %arg6)
                %14 = affine.apply #map1(%arg5, %arg6)
                %15 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %16 = tileflow.read_l1_buffer %0 at(%13, %14) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %17 = d2mtile.unary tile_transpose %16 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %1 at(%arg5, %arg6) epoch(%arg4) to %17 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                d2mtile.dst_release %15 : !d2mtile.dst_bank
              }
            }
            task.epoch_ready %1 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes() {
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
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %3 source_core(%16, %arg2) source_slice(%c9, %c0_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %4 source_core(%18, %arg2) source_slice(%c0_8, %c0_9) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_wait %3 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %4 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              task.epoch_acquire %7 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_acquire %8 epoch(%arg4) : !tileplan.l1_buffer
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %21 = arith.cmpi sge, %arg5, %c0_10 : index
                %22 = arith.cmpi slt, %arg5, %c1_11 : index
                %23 = arith.andi %21, %22 : i1
                %24 = arith.cmpi sge, %arg6, %c0_12 : index
                %25 = arith.cmpi slt, %arg6, %c10_13 : index
                %26 = arith.andi %24, %25 : i1
                %27 = arith.andi %23, %26 : i1
                %28 = affine.apply #map3(%arg6)
                %29 = tileflow.read_l1_buffer %3 at(%19, %28) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
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
                tileflow.write_l1_buffer %7 at(%arg5, %arg6) epoch(%arg4) to %36 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
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
                %48 = tileflow.read_l1_buffer %4 at(%20, %47) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
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
                tileflow.write_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) to %52 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
              task.epoch_ready %8 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_ready %7 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %3 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @dm1_0 engine(<dm1>) domain(<phase>) reads() writes() after [#task.dep<@compute_0, raw>] {
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
            task.epoch_wait %1 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %1 into %5 source_core(%arg1, %14) source_slice(%c0, %c0_3) epoch(%arg4) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %1 into %6 source_core(%arg1, %16) source_slice(%c0_5, %c9_6) epoch(%arg4) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_wait %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %6 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              task.epoch_acquire %9 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_acquire %10 epoch(%arg4) : !tileplan.l1_buffer
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
                %34 = tileflow.read_l1_buffer %5 at(%33, %18) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %35 = scf.if %32 -> (!d2mtile.tile<bf16>) {
                  scf.yield %34 : !d2mtile.tile<bf16>
                } else {
                  %53 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %54 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                  %55 = tileflow.read_l1_buffer %1 at(%53, %54) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %55 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %36 = tileflow.lane_copy %25 with %35 if(%17) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) to %36 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
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
                %50 = tileflow.read_l1_buffer %6 at(%49, %20) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %51 = scf.if %48 -> (!d2mtile.tile<bf16>) {
                  scf.yield %50 : !d2mtile.tile<bf16>
                } else {
                  %53 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %54 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                  %55 = tileflow.read_l1_buffer %1 at(%53, %54) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %55 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %52 = tileflow.lane_copy %41 with %51 if(%19) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) to %52 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
              task.epoch_ready %10 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_ready %9 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_free %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %1 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_0_1 engine(<compute>) domain(<phase>) reads(%3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %1 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full]) writes() {
            task.epoch_acquire %12 epoch(%arg4) : !tileplan.l1_buffer
            %13 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
            %14 = d2mtile.fill "2.000000e-01" -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
            %c0 = arith.constant 0 : index
            %c0_1 = arith.constant 0 : index
            tileflow.write_l1_buffer %12 at(%c0, %c0_1) epoch(%arg4) to %14 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
            task.epoch_ready %12 epoch(%arg4) : !tileplan.l1_buffer
            d2mtile.dst_release %13 : !d2mtile.dst_bank
            %c10_2 = arith.constant 10 : index
            %c10_3 = arith.constant 10 : index
            task.epoch_wait %12 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %2 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_2) axis <row> {
              task.epoch_wait %7 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_wait %8 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_wait %9 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_wait %10 epoch(%arg4) : !tileplan.l1_buffer
              tileplan.shard_for (%arg6) in (%c10_3) axis <col> {
                %15 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %16 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %17 = tileflow.read_l1_buffer %0 at(%15, %16) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %18 = tileflow.read_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %19 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %20 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %21 = d2mtile.binary tile_add %19, %18 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                %22 = d2mtile.unary tile_transpose %21 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                %23 = tileflow.read_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %24 = d2mtile.binary tile_add %17, %23 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                %25 = tileflow.read_l1_buffer %7 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %26 = d2mtile.binary tile_add %24, %25 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                %27 = d2mtile.binary tile_add %26, %22 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                %c0_4 = arith.constant 0 : index
                %c0_5 = arith.constant 0 : index
                %28 = tileflow.read_l1_buffer %12 at(%c0_4, %c0_5) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %29 = d2mtile.binary tile_mul %27, %28 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %2 at(%arg5, %arg6) epoch(%arg4) to %29 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                d2mtile.dst_release %20 : !d2mtile.dst_bank
              }
              task.epoch_free %10 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_free %9 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_free %8 epoch(%arg4) : !tileplan.l1_buffer
              task.epoch_free %7 epoch(%arg4) : !tileplan.l1_buffer
            }
            task.epoch_ready %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %12 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %0 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @dm0_1 engine(<dm0>) domain(<phase>) reads() writes() after [#task.dep<@compute_0_1, raw>] {
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
            task.epoch_acquire %0 epoch(%arg4) {task.prime_before_loop} : !tileplan.l1_buffer
            task.epoch_wait %2 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %13 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %14 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %15 = tileflow.read_l1_buffer %0 at(%13, %14) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %16 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %17 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %18 = tileflow.read_l1_buffer %2 at(%16, %17) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %19 = affine.apply #map10(%arg1, %arg5)
                %20 = affine.apply #map11()[%19]
                %21 = arith.cmpi sge, %20, %c0 : index
                %22 = arith.xori %21, %true : i1
                %23 = affine.apply #map12(%arg1, %arg5)
                %24 = affine.apply #map13()[%23]
                %25 = arith.cmpi sge, %24, %c0_3 : index
                %26 = arith.xori %25, %true_4 : i1
                %27 = arith.ori %22, %26 : i1
                %28 = affine.apply #map10(%arg2, %arg6)
                %29 = affine.apply #map11()[%28]
                %30 = arith.cmpi sge, %29, %c0_5 : index
                %31 = arith.xori %30, %true_6 : i1
                %32 = arith.ori %27, %31 : i1
                %33 = affine.apply #map12(%arg2, %arg6)
                %34 = affine.apply #map13()[%33]
                %35 = arith.cmpi sge, %34, %c0_7 : index
                %36 = arith.xori %35, %true_8 : i1
                %37 = arith.ori %32, %36 : i1
                scf.if %37 {
                  %38 = tileflow.lane_copy %18 with %15 if(%22) axis row src_lane 0 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                  %39 = tileflow.lane_copy %38 with %15 if(%26) axis row src_lane 31 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                  %40 = tileflow.lane_copy %39 with %15 if(%31) axis col src_lane 0 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                  %41 = tileflow.lane_copy %40 with %15 if(%36) axis col src_lane 31 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                  task.epoch_acquire %11 epoch(%arg4) : !tileplan.l1_buffer
                  tileflow.write_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) to %41 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                  task.epoch_ready %11 epoch(%arg4) : !tileplan.l1_buffer
                  task.epoch_wait %11 epoch(%arg4) : !tileplan.l1_buffer
                  %42 = tileflow.read_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %42 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                  task.epoch_free %11 epoch(%arg4) : !tileplan.l1_buffer
                } else {
                  tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %18 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
              }
            }
            task.epoch_free %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_ready %0 epoch(%arg4) : !tileplan.l1_buffer
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

