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
      %0 = tileplan.persistent_l1 elem f32 tile(32, 32) domain(10, 10) order(0) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<3200x3200xf32>) {
      ^bb0(%arg3: memref<3200x3200xf32>):
        %1 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) order(0) {frame_axis = #tileplan.axis<row_col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(2, 1) order(1) {frame_axis = #tileplan.axis<row>, task.frame_access = "replay", task.reuse = 1000 : i64, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(1, 2) order(2) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 10 : i64, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(1, 10) order(3) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(1, 10) order(4) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(10, 1) order(5) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<receive> elem f32 tile(32, 32) domain(10, 1) order(6) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) resident(1, 1) order(7) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) resident(1, 1) order(8) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) resident(1, 1) order(9) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %11 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) resident(1, 1) order(10) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %12 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) resident(1, 1) order(11) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %13 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) resident(1, 1) order(12) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %14 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) resident(1, 1) order(13) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %15 = tileplan.l1_buffer role<constructed> elem f32 tile(32, 32) domain(10, 10) resident(1, 1) order(14) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %16 = tileplan.l1_buffer role<intermediate> name "dst_spill_15" elem f32 tile(32, 32) domain(1, 1) order(15) {tt.dst_spill, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<3200x3200xf32>) writes(%arg3 : memref<3200x3200xf32>) {producer = "dm"}
        task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes(%2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) {
          %c1 = arith.constant 1 : index
          %c1_1 = arith.constant 1 : index
          task.epoch_acquire %2 : !tileplan.l1_buffer
          tileplan.shard_for (%arg4) in (%c1) axis <row> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %17 = affine.apply #map(%arg1, %c0_3)
            %18 = tileflow.row_mask symbols(%17) set #set invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %2 at(%c0, %c0_2) to %18 : !tileplan.l1_buffer to !tileflow.mask<f32>
            %c1_4 = arith.constant 1 : index
            %c0_5 = arith.constant 0 : index
            %c9 = arith.constant 9 : index
            %19 = affine.apply #map1(%arg1, %c9)
            %20 = tileflow.row_mask symbols(%19) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %2 at(%c1_4, %c0_5) to %20 : !tileplan.l1_buffer to !tileflow.mask<f32>
          }
          task.epoch_ready %2 : !tileplan.l1_buffer
          task.epoch_acquire %3 : !tileplan.l1_buffer
          tileplan.shard_for (%arg4) in (%c1_1) axis <col> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %17 = affine.apply #map(%arg2, %c0_3)
            %18 = tileflow.col_mask symbols(%17) set #set invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %3 at(%c0, %c0_2) to %18 : !tileplan.l1_buffer to !tileflow.mask<f32>
            %c0_4 = arith.constant 0 : index
            %c1_5 = arith.constant 1 : index
            %c9 = arith.constant 9 : index
            %19 = affine.apply #map1(%arg2, %c9)
            %20 = tileflow.col_mask symbols(%19) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<f32>
            tileflow.write_l1_buffer %3 at(%c0_4, %c1_5) to %20 : !tileplan.l1_buffer to !tileflow.mask<f32>
          }
          task.epoch_ready %3 : !tileplan.l1_buffer
        }
        tileplan.temporal_for (%arg4) in (0, 1000, 1) {
          task.group @dm0_0_1 engine(<dm0>) domain(<phase>) reads() writes() {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %true = arith.constant true
            %true_3 = arith.constant true
            %17 = arith.andi %true, %true_3 : i1
            %true_4 = arith.constant true
            %true_5 = arith.constant true
            %18 = arith.andi %true_4, %true_5 : i1
            %c-1 = arith.constant -1 : index
            %19 = arith.addi %arg1, %c-1 : index
            %c0 = arith.constant 0 : index
            %20 = arith.maxsi %19, %c0 : index
            %c9 = arith.constant 9 : index
            %c0_6 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %21 = arith.addi %arg1, %c1 : index
            %c9_7 = arith.constant 9 : index
            %22 = arith.minsi %21, %c9_7 : index
            %c0_8 = arith.constant 0 : index
            %c0_9 = arith.constant 0 : index
            %c1_10 = arith.constant 1 : index
            %23 = arith.addi %arg2, %c1_10 : index
            %c9_11 = arith.constant 9 : index
            %24 = arith.minsi %23, %c9_11 : index
            %c0_12 = arith.constant 0 : index
            %c0_13 = arith.constant 0 : index
            %c-1_14 = arith.constant -1 : index
            %25 = arith.addi %arg2, %c-1_14 : index
            %c0_15 = arith.constant 0 : index
            %26 = arith.maxsi %25, %c0_15 : index
            %c0_16 = arith.constant 0 : index
            %c9_17 = arith.constant 9 : index
            %c0_18 = arith.constant 0 : index
            %c1_19 = arith.constant 1 : index
            %c0_20 = arith.constant 0 : index
            %c10_21 = arith.constant 10 : index
            %27 = affine.apply #map2()
            %c9_22 = arith.constant 9 : index
            %c10_23 = arith.constant 10 : index
            %c0_24 = arith.constant 0 : index
            %c10_25 = arith.constant 10 : index
            %28 = affine.apply #map2()
            %true_26 = arith.constant true
            %true_27 = arith.constant true
            %29 = arith.andi %true_26, %true_27 : i1
            %true_28 = arith.constant true
            %true_29 = arith.constant true
            %30 = arith.andi %true_28, %true_29 : i1
            %true_30 = arith.constant true
            %true_31 = arith.constant true
            %true_32 = arith.constant true
            %true_33 = arith.constant true
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %4 source_core(%20, %arg2) source_slice(%c9, %c0_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %5 source_core(%22, %arg2) source_slice(%c0_8, %c0_9) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %6 source_core(%arg1, %24) source_slice(%c0_12, %c0_13) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %7 source_core(%arg1, %26) source_slice(%c0_16, %c9_17) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_wait %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %5 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %31 = arith.cmpi sge, %arg5, %c0_18 : index
                %32 = arith.cmpi slt, %arg5, %c1_19 : index
                %33 = arith.andi %31, %32 : i1
                %34 = arith.cmpi sge, %arg6, %c0_20 : index
                %35 = arith.cmpi slt, %arg6, %c10_21 : index
                %36 = arith.andi %34, %35 : i1
                %37 = arith.andi %33, %36 : i1
                %38 = affine.apply #map3(%arg6)
                %39 = tileflow.read_l1_buffer %4 at(%27, %38) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %40 = scf.if %37 -> (!d2mtile.tile<f32>) {
                  scf.yield %39 : !d2mtile.tile<f32>
                } else {
                  %75 = affine.apply #map4(%arg1, %arg2, %arg5, %arg6)
                  %76 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %77 = tileflow.read_l1_buffer %0 at(%75, %76) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %77 : !d2mtile.tile<f32>
                } {task.engine_label = "dm"}
                %41 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %42 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %43 = tileflow.read_l1_buffer %0 at(%41, %42) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %44 = tileflow.roll %43 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %45 = tileflow.lane_copy %44 with %43 if(%true_30) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                %46 = tileflow.lane_copy %45 with %40 if(%17) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                task.epoch_acquire %8 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) to %46 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                task.epoch_ready %8 epoch(%arg4) : !tileplan.l1_buffer
                %47 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %48 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %49 = tileflow.read_l1_buffer %0 at(%47, %48) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %50 = arith.cmpi sge, %arg5, %c9_22 : index
                %51 = arith.cmpi slt, %arg5, %c10_23 : index
                %52 = arith.andi %50, %51 : i1
                %53 = arith.cmpi sge, %arg6, %c0_24 : index
                %54 = arith.cmpi slt, %arg6, %c10_25 : index
                %55 = arith.andi %53, %54 : i1
                %56 = arith.andi %52, %55 : i1
                %57 = affine.apply #map3(%arg6)
                %58 = tileflow.read_l1_buffer %5 at(%28, %57) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %59 = scf.if %56 -> (!d2mtile.tile<f32>) {
                  scf.yield %58 : !d2mtile.tile<f32>
                } else {
                  %75 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                  %76 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %77 = tileflow.read_l1_buffer %0 at(%75, %76) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %77 : !d2mtile.tile<f32>
                } {task.engine_label = "dm"}
                %60 = tileflow.roll %49 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %61 = tileflow.lane_copy %60 with %49 if(%true_31) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                %62 = tileflow.lane_copy %61 with %59 if(%18) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                task.epoch_acquire %9 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) to %62 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                task.epoch_ready %9 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %10 epoch(%arg4) : !tileplan.l1_buffer
                %63 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %64 = tileflow.roll %63 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %65 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %66 = tileflow.lane_copy %64 with %65 if(%true_32) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                task.epoch_wait %11 epoch(%arg4) : !tileplan.l1_buffer
                %67 = tileflow.read_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %68 = tileflow.lane_copy %66 with %67 if(%29) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                task.epoch_acquire %12 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) to %68 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                task.epoch_ready %12 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %11 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %10 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %13 epoch(%arg4) : !tileplan.l1_buffer
                %69 = tileflow.read_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %70 = tileflow.roll %69 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<f32>
                %71 = tileflow.read_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %72 = tileflow.lane_copy %70 with %71 if(%true_33) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                task.epoch_wait %14 epoch(%arg4) : !tileplan.l1_buffer
                %73 = tileflow.read_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %74 = tileflow.lane_copy %72 with %73 if(%30) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
                task.epoch_acquire %15 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %15 at(%arg5, %arg6) epoch(%arg4) to %74 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                task.epoch_ready %15 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %14 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %13 epoch(%arg4) : !tileplan.l1_buffer
              }
            }
            task.epoch_free %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %4 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_0 engine(<compute>) domain(<phase>) reads(%4 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) writes() after [#task.dep<@dm0_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %c10_3 = arith.constant 10 : index
            %c9 = arith.constant 9 : index
            %c10_4 = arith.constant 10 : index
            %17 = affine.apply #map2()
            %c0_5 = arith.constant 0 : index
            %c10_6 = arith.constant 10 : index
            %c0_7 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %18 = affine.apply #map2()
            %c0_8 = arith.constant 0 : index
            %c0_9 = arith.constant 0 : index
            %c0_10 = arith.constant 0 : index
            %c0_11 = arith.constant 0 : index
            task.epoch_acquire %16 epoch(%arg4) : !tileplan.l1_buffer
            %19 = d2mtile.dst_acquire capacity 4 : !d2mtile.dst_bank
            %20 = d2mtile.fill "2.000000e-01" -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<f32>
            %c0_12 = arith.constant 0 : index
            %c0_13 = arith.constant 0 : index
            tileflow.write_l1_buffer %16 at(%c0_12, %c0_13) epoch(%arg4) to %20 : !tileplan.l1_buffer to !d2mtile.tile<f32>
            task.epoch_ready %16 epoch(%arg4) : !tileplan.l1_buffer
            d2mtile.dst_release %19 : !d2mtile.dst_bank
            task.epoch_wait %2 : !tileplan.l1_buffer
            task.epoch_wait %3 : !tileplan.l1_buffer
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %7 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %16 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %1 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %21 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %22 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                task.epoch_acquire %10 epoch(%arg4) : !tileplan.l1_buffer
                %23 = d2mtile.dst_acquire capacity 4 : !d2mtile.dst_bank
                %24 = tileflow.read_l1_buffer %0 at(%21, %22) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %25 = d2mtile.unary tile_transpose %24 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) to %25 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                task.epoch_ready %10 epoch(%arg4) : !tileplan.l1_buffer
                d2mtile.dst_release %23 : !d2mtile.dst_bank
                task.epoch_acquire %11 epoch(%arg4) : !tileplan.l1_buffer
                %26 = d2mtile.dst_acquire capacity 4 : !d2mtile.dst_bank
                %27 = affine.apply #map3(%arg5)
                %28 = tileflow.read_l1_buffer %6 at(%27, %17) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %29 = arith.cmpi sge, %arg5, %c0 : index
                %30 = arith.cmpi slt, %arg5, %c10_3 : index
                %31 = arith.andi %29, %30 : i1
                %32 = arith.cmpi sge, %arg6, %c9 : index
                %33 = arith.cmpi slt, %arg6, %c10_4 : index
                %34 = arith.andi %32, %33 : i1
                %35 = arith.andi %31, %34 : i1
                %36 = scf.if %35 -> (!d2mtile.tile<f32>) {
                  scf.yield %28 : !d2mtile.tile<f32>
                } else {
                  %92 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %93 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                  %94 = tileflow.read_l1_buffer %0 at(%92, %93) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %94 : !d2mtile.tile<f32>
                } {task.engine_label = "compute", tt.dst_slot = 0 : i64}
                %37 = d2mtile.unary tile_transpose %36 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) to %37 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                task.epoch_ready %11 epoch(%arg4) : !tileplan.l1_buffer
                d2mtile.dst_release %26 : !d2mtile.dst_bank
                task.epoch_acquire %13 epoch(%arg4) : !tileplan.l1_buffer
                %38 = d2mtile.dst_acquire capacity 4 : !d2mtile.dst_bank
                %39 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %40 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %41 = tileflow.read_l1_buffer %0 at(%39, %40) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %42 = d2mtile.unary tile_transpose %41 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) to %42 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                task.epoch_ready %13 epoch(%arg4) : !tileplan.l1_buffer
                d2mtile.dst_release %38 : !d2mtile.dst_bank
                task.epoch_acquire %14 epoch(%arg4) : !tileplan.l1_buffer
                %43 = d2mtile.dst_acquire capacity 4 : !d2mtile.dst_bank
                %44 = affine.apply #map3(%arg5)
                %45 = tileflow.read_l1_buffer %7 at(%44, %18) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %46 = arith.cmpi sge, %arg5, %c0_5 : index
                %47 = arith.cmpi slt, %arg5, %c10_6 : index
                %48 = arith.andi %46, %47 : i1
                %49 = arith.cmpi sge, %arg6, %c0_7 : index
                %50 = arith.cmpi slt, %arg6, %c1 : index
                %51 = arith.andi %49, %50 : i1
                %52 = arith.andi %48, %51 : i1
                %53 = scf.if %52 -> (!d2mtile.tile<f32>) {
                  scf.yield %45 : !d2mtile.tile<f32>
                } else {
                  %92 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %93 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                  %94 = tileflow.read_l1_buffer %0 at(%92, %93) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                  scf.yield %94 : !d2mtile.tile<f32>
                } {task.engine_label = "compute", tt.dst_slot = 0 : i64}
                %54 = d2mtile.unary tile_transpose %53 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) to %54 : !tileplan.l1_buffer to !d2mtile.tile<f32>
                task.epoch_ready %14 epoch(%arg4) : !tileplan.l1_buffer
                d2mtile.dst_release %43 : !d2mtile.dst_bank
                %55 = d2mtile.dst_acquire capacity 4 : !d2mtile.dst_bank
                %56 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %57 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %58 = tileflow.read_l1_buffer %0 at(%56, %57) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                task.epoch_wait %12 epoch(%arg4) : !tileplan.l1_buffer
                %59 = tileflow.read_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                task.epoch_wait %15 epoch(%arg4) : !tileplan.l1_buffer
                %60 = tileflow.read_l1_buffer %15 at(%arg5, %arg6) epoch(%arg4) {tt.dst_slot = 2 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %61 = d2mtile.binary tile_add %60, %59 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<f32>
                task.epoch_free %15 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %12 epoch(%arg4) : !tileplan.l1_buffer
                %62 = d2mtile.unary tile_transpose %61 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<f32>
                task.epoch_wait %9 epoch(%arg4) : !tileplan.l1_buffer
                %63 = tileflow.read_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %64 = d2mtile.binary tile_add %58, %63 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 3 : i64} : !d2mtile.tile<f32>
                task.epoch_free %9 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %8 epoch(%arg4) : !tileplan.l1_buffer
                %65 = tileflow.read_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %66 = d2mtile.binary tile_add %64, %65 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 3 : i64} : !d2mtile.tile<f32>
                task.epoch_free %8 epoch(%arg4) : !tileplan.l1_buffer
                %67 = d2mtile.binary tile_add %66, %62 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 3 : i64} : !d2mtile.tile<f32>
                %c0_14 = arith.constant 0 : index
                %c0_15 = arith.constant 0 : index
                %68 = tileflow.read_l1_buffer %16 at(%c0_14, %c0_15) epoch(%arg4) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                %69 = d2mtile.binary tile_mul %67, %68 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 3 : i64} : !d2mtile.tile<f32>
                %c9_16 = arith.constant 9 : index
                %70 = arith.cmpi eq, %arg5, %c9_16 : index
                %c1_17 = arith.constant 1 : index
                %c0_18 = arith.constant 0 : index
                %71 = arith.select %70, %c1_17, %c0_18 : index
                %c0_19 = arith.constant 0 : index
                %72 = tileflow.read_l1_buffer %2 at(%71, %c0_19) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !tileflow.mask<f32>
                %73 = affine.apply #map(%arg1, %arg5)
                %74 = affine.apply #map10()[%73]
                %75 = arith.cmpi slt, %74, %c0_8 : index
                %76 = affine.apply #map1(%arg1, %arg5)
                %77 = affine.apply #map11()[%76]
                %78 = arith.cmpi slt, %77, %c0_9 : index
                %79 = arith.ori %75, %78 : i1
                %80 = tileflow.compute_select %69 with %58 active(%72 : !tileflow.mask<f32>) if(%79) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>, tt.dst_slot = 3 : i64} : !d2mtile.tile<f32>
                %c0_20 = arith.constant 0 : index
                %c9_21 = arith.constant 9 : index
                %81 = arith.cmpi eq, %arg6, %c9_21 : index
                %c1_22 = arith.constant 1 : index
                %c0_23 = arith.constant 0 : index
                %82 = arith.select %81, %c1_22, %c0_23 : index
                %83 = tileflow.read_l1_buffer %3 at(%c0_20, %82) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !tileflow.mask<f32>
                %84 = affine.apply #map(%arg2, %arg6)
                %85 = affine.apply #map10()[%84]
                %86 = arith.cmpi slt, %85, %c0_10 : index
                %87 = affine.apply #map1(%arg2, %arg6)
                %88 = affine.apply #map11()[%87]
                %89 = arith.cmpi slt, %88, %c0_11 : index
                %90 = arith.ori %86, %89 : i1
                %91 = tileflow.compute_select %80 with %58 active(%83 : !tileflow.mask<f32>) if(%90) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>, tt.dst_slot = 3 : i64} : !d2mtile.tile<f32>
                tileflow.write_l1_buffer %1 at(%arg5, %arg6) epoch(%arg4) to %91 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
                d2mtile.dst_release %55 : !d2mtile.dst_bank
              }
            }
            task.epoch_ready %1 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %16 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %7 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %0 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @dm0_3 engine(<dm0>) domain(<phase>) reads() writes() after [#task.dep<@compute_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            task.epoch_acquire %0 epoch(%arg4) {task.prime_before_loop} : !tileplan.l1_buffer
            task.epoch_wait %1 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %17 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %18 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %19 = tileflow.read_l1_buffer %1 at(%17, %18) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %19 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<f32>
              }
            }
            task.epoch_free %1 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_ready %0 epoch(%arg4) : !tileplan.l1_buffer
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

