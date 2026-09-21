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
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(10, 10) order(0) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
      %1 = tileplan.persistent_l1 name "frame.converted" elem bf16 tile(32, 32) domain(10, 10) order(1) {tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<3200x3200xbf16>) {
      ^bb0(%arg3: memref<3200x3200xbf16>):
        %2 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(0) {frame_axis = #tileplan.axis<row_col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(2, 1) order(1) {frame_axis = #tileplan.axis<row>, task.frame_access = "replay", task.reuse = 1000 : i64, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 2) order(2) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 10 : i64, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(3) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(4) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(5) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(6) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(7) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(8) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %11 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(9) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %12 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(10) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.mailbox_depth = 2 : i64} : !tileplan.l1_buffer
        %13 = tileplan.l1_buffer role<intermediate> name "dst_spill_11" elem bf16 tile(32, 32) domain(1, 1) order(11) {tt.dst_spill, tt.mailbox_depth = 1 : i64} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<3200x3200xbf16>) writes(%arg3 : memref<3200x3200xbf16>) {producer = "dm"}
        tileplan.bind_persistent %1 : !tileplan.l1_buffer reads() writes() {producer = "compute"}
        task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes(%3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full]) {
          %c1 = arith.constant 1 : index
          %c1_1 = arith.constant 1 : index
          task.epoch_acquire %3 : !tileplan.l1_buffer
          tileplan.shard_for (%arg4) in (%c1) axis <row> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %14 = affine.apply #map(%arg1, %c0_3)
            %15 = tileflow.row_mask symbols(%14) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %3 at(%c0, %c0_2) to %15 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            %c1_4 = arith.constant 1 : index
            %c0_5 = arith.constant 0 : index
            %c9 = arith.constant 9 : index
            %16 = affine.apply #map1(%arg1, %c9)
            %17 = tileflow.row_mask symbols(%16) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %3 at(%c1_4, %c0_5) to %17 : !tileplan.l1_buffer to !tileflow.mask<bf16>
          }
          task.epoch_ready %3 : !tileplan.l1_buffer
          task.epoch_acquire %4 : !tileplan.l1_buffer
          tileplan.shard_for (%arg4) in (%c1_1) axis <col> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %14 = affine.apply #map(%arg2, %c0_3)
            %15 = tileflow.col_mask symbols(%14) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %4 at(%c0, %c0_2) to %15 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            %c0_4 = arith.constant 0 : index
            %c1_5 = arith.constant 1 : index
            %c9 = arith.constant 9 : index
            %16 = affine.apply #map1(%arg2, %c9)
            %17 = tileflow.col_mask symbols(%16) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %4 at(%c0_4, %c1_5) to %17 : !tileplan.l1_buffer to !tileflow.mask<bf16>
          }
          task.epoch_ready %4 : !tileplan.l1_buffer
        }
        tileplan.temporal_for (%arg4) in (0, 1000, 1) {
          task.group @dm0_0_1 engine(<dm0>) domain(<phase>) reads() writes() {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %true = arith.constant true
            %true_3 = arith.constant true
            %14 = arith.andi %true, %true_3 : i1
            %true_4 = arith.constant true
            %true_5 = arith.constant true
            %15 = arith.andi %true_4, %true_5 : i1
            %c-1 = arith.constant -1 : index
            %16 = arith.addi %arg1, %c-1 : index
            %c0 = arith.constant 0 : index
            %17 = arith.maxsi %16, %c0 : index
            %c9 = arith.constant 9 : index
            %c0_6 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %18 = arith.addi %arg1, %c1 : index
            %c9_7 = arith.constant 9 : index
            %19 = arith.minsi %18, %c9_7 : index
            %c0_8 = arith.constant 0 : index
            %c0_9 = arith.constant 0 : index
            %c0_10 = arith.constant 0 : index
            %c1_11 = arith.constant 1 : index
            %c0_12 = arith.constant 0 : index
            %c10_13 = arith.constant 10 : index
            %20 = affine.apply #map2()
            %c9_14 = arith.constant 9 : index
            %c10_15 = arith.constant 10 : index
            %c0_16 = arith.constant 0 : index
            %c10_17 = arith.constant 10 : index
            %21 = affine.apply #map2()
            %true_18 = arith.constant true
            %true_19 = arith.constant true
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %5 source_core(%17, %arg2) source_slice(%c9, %c0_6) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %6 source_core(%19, %arg2) source_slice(%c0_8, %c0_9) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_wait %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %6 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %22 = arith.cmpi sge, %arg5, %c0_10 : index
                %23 = arith.cmpi slt, %arg5, %c1_11 : index
                %24 = arith.andi %22, %23 : i1
                %25 = arith.cmpi sge, %arg6, %c0_12 : index
                %26 = arith.cmpi slt, %arg6, %c10_13 : index
                %27 = arith.andi %25, %26 : i1
                %28 = arith.andi %24, %27 : i1
                %29 = affine.apply #map3(%arg6)
                %30 = tileflow.read_l1_buffer %5 at(%20, %29) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %31 = scf.if %28 -> (!d2mtile.tile<bf16>) {
                  scf.yield %30 : !d2mtile.tile<bf16>
                } else {
                  %54 = affine.apply #map4(%arg1, %arg2, %arg5, %arg6)
                  %55 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %56 = tileflow.read_l1_buffer %0 at(%54, %55) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %56 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %32 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %33 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %34 = tileflow.read_l1_buffer %0 at(%32, %33) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %35 = tileflow.roll %34 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %36 = tileflow.lane_copy %35 with %34 if(%true_18) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %37 = tileflow.lane_copy %36 with %31 if(%14) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %9 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) to %37 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %9 epoch(%arg4) : !tileplan.l1_buffer
                %38 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %39 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %40 = tileflow.read_l1_buffer %0 at(%38, %39) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %41 = arith.cmpi sge, %arg5, %c9_14 : index
                %42 = arith.cmpi slt, %arg5, %c10_15 : index
                %43 = arith.andi %41, %42 : i1
                %44 = arith.cmpi sge, %arg6, %c0_16 : index
                %45 = arith.cmpi slt, %arg6, %c10_17 : index
                %46 = arith.andi %44, %45 : i1
                %47 = arith.andi %43, %46 : i1
                %48 = affine.apply #map3(%arg6)
                %49 = tileflow.read_l1_buffer %6 at(%21, %48) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %50 = scf.if %47 -> (!d2mtile.tile<bf16>) {
                  scf.yield %49 : !d2mtile.tile<bf16>
                } else {
                  %54 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                  %55 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %56 = tileflow.read_l1_buffer %0 at(%54, %55) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %56 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %51 = tileflow.roll %40 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %52 = tileflow.lane_copy %51 with %40 if(%true_19) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %53 = tileflow.lane_copy %52 with %50 if(%15) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %10 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) to %53 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %10 epoch(%arg4) : !tileplan.l1_buffer
              }
            }
            task.epoch_free %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %5 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_0 engine(<compute>) domain(<phase>) reads() writes() {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %1 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %14 = affine.apply #map8(%arg5, %arg6)
                %15 = affine.apply #map9(%arg5, %arg6)
                %16 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %17 = tileflow.read_l1_buffer %0 at(%14, %15) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %18 = d2mtile.unary tile_transpose %17 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %1 at(%arg5, %arg6) epoch(%arg4) to %18 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                d2mtile.dst_release %16 : !d2mtile.dst_bank
              }
            }
            task.epoch_ready %1 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @dm1_0 engine(<dm1>) domain(<phase>) reads() writes() after [#task.dep<@compute_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c1 = arith.constant 1 : index
            %14 = arith.addi %arg2, %c1 : index
            %c9 = arith.constant 9 : index
            %15 = arith.minsi %14, %c9 : index
            %c0 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %c-1 = arith.constant -1 : index
            %16 = arith.addi %arg2, %c-1 : index
            %c0_4 = arith.constant 0 : index
            %17 = arith.maxsi %16, %c0_4 : index
            %c0_5 = arith.constant 0 : index
            %c9_6 = arith.constant 9 : index
            %true = arith.constant true
            %true_7 = arith.constant true
            %18 = arith.andi %true, %true_7 : i1
            %c0_8 = arith.constant 0 : index
            %c10_9 = arith.constant 10 : index
            %c9_10 = arith.constant 9 : index
            %c10_11 = arith.constant 10 : index
            %19 = affine.apply #map2()
            %true_12 = arith.constant true
            %true_13 = arith.constant true
            %20 = arith.andi %true_12, %true_13 : i1
            %c0_14 = arith.constant 0 : index
            %c10_15 = arith.constant 10 : index
            %c0_16 = arith.constant 0 : index
            %c1_17 = arith.constant 1 : index
            %21 = affine.apply #map2()
            %true_18 = arith.constant true
            %true_19 = arith.constant true
            task.epoch_wait %1 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %1 into %7 source_core(%arg1, %15) source_slice(%c0, %c0_3) epoch(%arg4) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %1 into %8 source_core(%arg1, %17) source_slice(%c0_5, %c9_6) epoch(%arg4) {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_wait %7 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %8 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %22 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %23 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %24 = tileflow.read_l1_buffer %1 at(%22, %23) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %25 = tileflow.roll %24 axis row by 1 {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %26 = tileflow.lane_copy %25 with %24 if(%true_18) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %27 = arith.cmpi sge, %arg5, %c0_8 : index
                %28 = arith.cmpi slt, %arg5, %c10_9 : index
                %29 = arith.andi %27, %28 : i1
                %30 = arith.cmpi sge, %arg6, %c9_10 : index
                %31 = arith.cmpi slt, %arg6, %c10_11 : index
                %32 = arith.andi %30, %31 : i1
                %33 = arith.andi %29, %32 : i1
                %34 = affine.apply #map3(%arg5)
                %35 = tileflow.read_l1_buffer %7 at(%34, %19) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %36 = scf.if %33 -> (!d2mtile.tile<bf16>) {
                  scf.yield %35 : !d2mtile.tile<bf16>
                } else {
                  %54 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %55 = affine.apply #map10(%arg1, %arg2, %arg5, %arg6)
                  %56 = tileflow.read_l1_buffer %1 at(%54, %55) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %56 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %37 = tileflow.lane_copy %26 with %36 if(%18) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %11 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) to %37 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %11 epoch(%arg4) : !tileplan.l1_buffer
                %38 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %39 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %40 = tileflow.read_l1_buffer %1 at(%38, %39) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %41 = tileflow.roll %40 axis row by -1 {task.dm_lane = #task.engine<dm1>, task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %42 = tileflow.lane_copy %41 with %40 if(%true_19) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %43 = arith.cmpi sge, %arg5, %c0_14 : index
                %44 = arith.cmpi slt, %arg5, %c10_15 : index
                %45 = arith.andi %43, %44 : i1
                %46 = arith.cmpi sge, %arg6, %c0_16 : index
                %47 = arith.cmpi slt, %arg6, %c1_17 : index
                %48 = arith.andi %46, %47 : i1
                %49 = arith.andi %45, %48 : i1
                %50 = affine.apply #map3(%arg5)
                %51 = tileflow.read_l1_buffer %8 at(%50, %21) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %52 = scf.if %49 -> (!d2mtile.tile<bf16>) {
                  scf.yield %51 : !d2mtile.tile<bf16>
                } else {
                  %54 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %55 = affine.apply #map11(%arg1, %arg2, %arg5, %arg6)
                  %56 = tileflow.read_l1_buffer %1 at(%54, %55) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %56 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %53 = tileflow.lane_copy %42 with %52 if(%20) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %12 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) to %53 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %12 epoch(%arg4) : !tileplan.l1_buffer
              }
            }
            task.epoch_free %8 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %7 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %1 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_0_1 engine(<compute>) domain(<phase>) reads(%5 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %1 : !tileplan.l1_buffer [full], %8 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full], %4 : !tileplan.l1_buffer [full]) writes() after [#task.dep<@dm0_0, raw>] {
            task.epoch_acquire %13 epoch(%arg4) : !tileplan.l1_buffer
            %14 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
            %15 = d2mtile.fill "2.000000e-01" -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
            %c0 = arith.constant 0 : index
            %c0_1 = arith.constant 0 : index
            tileflow.write_l1_buffer %13 at(%c0, %c0_1) epoch(%arg4) to %15 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
            task.epoch_ready %13 epoch(%arg4) : !tileplan.l1_buffer
            d2mtile.dst_release %14 : !d2mtile.dst_bank
            %c10_2 = arith.constant 10 : index
            %c10_3 = arith.constant 10 : index
            %c0_4 = arith.constant 0 : index
            %c0_5 = arith.constant 0 : index
            %c0_6 = arith.constant 0 : index
            %c0_7 = arith.constant 0 : index
            task.epoch_wait %3 : !tileplan.l1_buffer
            task.epoch_wait %4 : !tileplan.l1_buffer
            task.epoch_wait %13 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %2 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_2) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_3) axis <col> {
                %16 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %17 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %18 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %19 = tileflow.read_l1_buffer %0 at(%16, %17) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                task.epoch_wait %11 epoch(%arg4) : !tileplan.l1_buffer
                %20 = tileflow.read_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                task.epoch_wait %12 epoch(%arg4) : !tileplan.l1_buffer
                %21 = tileflow.read_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %22 = d2mtile.binary tile_add %21, %20 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %12 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %11 epoch(%arg4) : !tileplan.l1_buffer
                %23 = d2mtile.unary tile_transpose %22 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                task.epoch_wait %10 epoch(%arg4) : !tileplan.l1_buffer
                %24 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %25 = d2mtile.binary tile_add %19, %24 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %10 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %9 epoch(%arg4) : !tileplan.l1_buffer
                %26 = tileflow.read_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %27 = d2mtile.binary tile_add %25, %26 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %9 epoch(%arg4) : !tileplan.l1_buffer
                %28 = d2mtile.binary tile_add %27, %23 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                %c0_8 = arith.constant 0 : index
                %c0_9 = arith.constant 0 : index
                %29 = tileflow.read_l1_buffer %13 at(%c0_8, %c0_9) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %30 = d2mtile.binary tile_mul %28, %29 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                %c9 = arith.constant 9 : index
                %31 = arith.cmpi eq, %arg5, %c9 : index
                %c1 = arith.constant 1 : index
                %c0_10 = arith.constant 0 : index
                %32 = arith.select %31, %c1, %c0_10 : index
                %c0_11 = arith.constant 0 : index
                %33 = tileflow.read_l1_buffer %3 at(%32, %c0_11) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %34 = affine.apply #map(%arg1, %arg5)
                %35 = affine.apply #map12()[%34]
                %36 = arith.cmpi slt, %35, %c0_4 : index
                %37 = affine.apply #map1(%arg1, %arg5)
                %38 = affine.apply #map13()[%37]
                %39 = arith.cmpi slt, %38, %c0_5 : index
                %40 = arith.ori %36, %39 : i1
                %41 = tileflow.compute_select %30 with %19 active(%33 : !tileflow.mask<bf16>) if(%40) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>, tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                %c0_12 = arith.constant 0 : index
                %c9_13 = arith.constant 9 : index
                %42 = arith.cmpi eq, %arg6, %c9_13 : index
                %c1_14 = arith.constant 1 : index
                %c0_15 = arith.constant 0 : index
                %43 = arith.select %42, %c1_14, %c0_15 : index
                %44 = tileflow.read_l1_buffer %4 at(%c0_12, %43) {tt.dst_slot = 1 : i64} : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %45 = affine.apply #map(%arg2, %arg6)
                %46 = affine.apply #map12()[%45]
                %47 = arith.cmpi slt, %46, %c0_6 : index
                %48 = affine.apply #map1(%arg2, %arg6)
                %49 = affine.apply #map13()[%48]
                %50 = arith.cmpi slt, %49, %c0_7 : index
                %51 = arith.ori %47, %50 : i1
                %52 = tileflow.compute_select %41 with %19 active(%44 : !tileflow.mask<bf16>) if(%51) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>, tt.dst_slot = 2 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %2 at(%arg5, %arg6) epoch(%arg4) to %52 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                d2mtile.dst_release %18 : !d2mtile.dst_bank
              }
            }
            task.epoch_ready %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %13 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %0 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @dm0_2 engine(<dm0>) domain(<phase>) reads() writes() after [#task.dep<@compute_0_1, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            task.epoch_acquire %0 epoch(%arg4) {task.prime_before_loop} : !tileplan.l1_buffer
            task.epoch_wait %2 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %14 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %15 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %16 = tileflow.read_l1_buffer %2 at(%14, %15) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %16 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
              }
            }
            task.epoch_free %2 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_ready %0 epoch(%arg4) : !tileplan.l1_buffer
          }
        } {epoch_id = 0 : i64}
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

