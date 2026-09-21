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
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute, tt.l1_peak = 542720 : i64, tt.l1_persistent_base = 337920 : i64, tt.l1_pool = 337920 : i64} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg1, %arg2) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(10, 10) order(0) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tt.frame_pages = 100 : i64, tt.l1_offset = 337920 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 204800 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "persistent"} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins() outs(%arg0 : memref<3200x3200xbf16>) {
      ^bb0(%arg3: memref<3200x3200xbf16>):
        %1 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) order(0) {frame_axis = #tileplan.axis<row_col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<resident>, tt.cb_id = 0 : i64, tt.frame_pages = 100 : i64, tt.l1_offset = 0 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 204800 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %2 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(2, 1) order(1) {frame_axis = #tileplan.axis<row>, task.frame_access = "replay", task.reuse = 1000 : i64, task.slot_access = #task.slot_access<resident>, tt.cb_id = 1 : i64, tt.frame_pages = 2 : i64, tt.l1_offset = 204800 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(1, 2) order(2) {frame_axis = #tileplan.axis<col>, task.frame_access = "replay", task.reuse = 10 : i64, task.slot_access = #task.slot_access<resident>, tt.cb_id = 2 : i64, tt.frame_pages = 2 : i64, tt.l1_offset = 208896 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %4 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(3) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 3 : i64, tt.frame_pages = 10 : i64, tt.l1_offset = 212992 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 20480 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %5 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 10) order(4) {frame_axis = #tileplan.axis<col>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 4 : i64, tt.frame_pages = 10 : i64, tt.l1_offset = 233472 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 20480 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %6 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(5) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 5 : i64, tt.frame_pages = 10 : i64, tt.l1_offset = 253952 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 20480 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %7 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(10, 1) order(6) {frame_axis = #tileplan.axis<row>, task.delivery_initialized, task.slot_access = #task.slot_access<resident>, tt.cb_id = 6 : i64, tt.frame_pages = 10 : i64, tt.l1_offset = 274432 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 20480 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %8 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(7) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 7 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 294912 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %9 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(8) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 8 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 299008 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %10 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(9) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 9 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 303104 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %11 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(10) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 10 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 307200 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %12 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(11) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 11 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 311296 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %13 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(12) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 12 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 315392 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %14 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(13) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 13 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 319488 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %15 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(14) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 14 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 323584 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %16 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(15) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 15 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 327680 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %17 = tileplan.l1_buffer role<constructed> elem bf16 tile(32, 32) domain(10, 10) resident(1, 1) order(16) {frame_axis = #tileplan.axis<row_col>, task.drain_axis = #tileplan.axis<col>, task.fill_axis = #tileplan.axis<col>, task.slot_access = #task.slot_access<single_pass>, task.stream = #tileplan.axis<col>, tt.cb_id = 16 : i64, tt.frame_pages = 1 : i64, tt.l1_offset = 331776 : i64, tt.mailbox_depth = 2 : i64, tt.storage_bytes = 4096 : i64, tt.storage_depth = 2 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        %18 = tileplan.l1_buffer role<intermediate> name "dst_spill_17" elem bf16 tile(32, 32) domain(1, 1) order(17) {tt.cb_id = 17 : i64, tt.dst_spill, tt.frame_pages = 1 : i64, tt.l1_offset = 335872 : i64, tt.mailbox_depth = 1 : i64, tt.storage_bytes = 2048 : i64, tt.storage_depth = 1 : i64, tt.storage_kind = "addressed_mailbox"} : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg3 : memref<3200x3200xbf16>) writes(%arg3 : memref<3200x3200xbf16>) {producer = "dm", tt.cb_id = 18 : i64}
        task.group @dm0_0 engine(<dm0>) domain(<phase>) reads() writes(%2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) {
          %c1 = arith.constant 1 : index
          %c1_1 = arith.constant 1 : index
          task.epoch_acquire %2 : !tileplan.l1_buffer
          tileplan.shard_for (%arg4) in (%c1) axis <row> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %19 = affine.apply #map(%arg1, %c0_3)
            %20 = tileflow.row_mask symbols(%19) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %2 at(%c0, %c0_2) to %20 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            %c1_4 = arith.constant 1 : index
            %c0_5 = arith.constant 0 : index
            %c9 = arith.constant 9 : index
            %21 = affine.apply #map1(%arg1, %c9)
            %22 = tileflow.row_mask symbols(%21) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %2 at(%c1_4, %c0_5) to %22 : !tileplan.l1_buffer to !tileflow.mask<bf16>
          }
          task.epoch_ready %2 : !tileplan.l1_buffer
          task.epoch_acquire %3 : !tileplan.l1_buffer
          tileplan.shard_for (%arg4) in (%c1_1) axis <col> {
            %c0 = arith.constant 0 : index
            %c0_2 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %19 = affine.apply #map(%arg2, %c0_3)
            %20 = tileflow.col_mask symbols(%19) set #set invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %3 at(%c0, %c0_2) to %20 : !tileplan.l1_buffer to !tileflow.mask<bf16>
            %c0_4 = arith.constant 0 : index
            %c1_5 = arith.constant 1 : index
            %c9 = arith.constant 9 : index
            %21 = affine.apply #map1(%arg2, %c9)
            %22 = tileflow.col_mask symbols(%21) set #set1 invert {task.engine_label = "dm"} : !tileflow.mask<bf16>
            tileflow.write_l1_buffer %3 at(%c0_4, %c1_5) to %22 : !tileplan.l1_buffer to !tileflow.mask<bf16>
          }
          task.epoch_ready %3 : !tileplan.l1_buffer
        }
        tileplan.temporal_for (%arg4) in (0, 1000, 1) {
          task.group @dm0_0_1 engine(<dm0>) domain(<phase>) reads() writes() {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %true = arith.constant true
            %true_3 = arith.constant true
            %19 = arith.andi %true, %true_3 : i1
            %true_4 = arith.constant true
            %true_5 = arith.constant true
            %20 = arith.andi %true_4, %true_5 : i1
            %true_6 = arith.constant true
            %true_7 = arith.constant true
            %21 = arith.andi %true_6, %true_7 : i1
            %true_8 = arith.constant true
            %true_9 = arith.constant true
            %22 = arith.andi %true_8, %true_9 : i1
            %c-1 = arith.constant -1 : index
            %23 = arith.addi %arg1, %c-1 : index
            %c0 = arith.constant 0 : index
            %24 = arith.maxsi %23, %c0 : index
            %c9 = arith.constant 9 : index
            %c0_10 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %25 = arith.addi %arg1, %c1 : index
            %c9_11 = arith.constant 9 : index
            %26 = arith.minsi %25, %c9_11 : index
            %c0_12 = arith.constant 0 : index
            %c0_13 = arith.constant 0 : index
            %c1_14 = arith.constant 1 : index
            %27 = arith.addi %arg2, %c1_14 : index
            %c9_15 = arith.constant 9 : index
            %28 = arith.minsi %27, %c9_15 : index
            %c0_16 = arith.constant 0 : index
            %c0_17 = arith.constant 0 : index
            %c-1_18 = arith.constant -1 : index
            %29 = arith.addi %arg2, %c-1_18 : index
            %c0_19 = arith.constant 0 : index
            %30 = arith.maxsi %29, %c0_19 : index
            %c0_20 = arith.constant 0 : index
            %c9_21 = arith.constant 9 : index
            %c0_22 = arith.constant 0 : index
            %c1_23 = arith.constant 1 : index
            %c0_24 = arith.constant 0 : index
            %c10_25 = arith.constant 10 : index
            %31 = affine.apply #map2()
            %c9_26 = arith.constant 9 : index
            %c10_27 = arith.constant 10 : index
            %c0_28 = arith.constant 0 : index
            %c10_29 = arith.constant 10 : index
            %32 = affine.apply #map2()
            %c0_30 = arith.constant 0 : index
            %c10_31 = arith.constant 10 : index
            %c9_32 = arith.constant 9 : index
            %c10_33 = arith.constant 10 : index
            %33 = affine.apply #map2()
            %c0_34 = arith.constant 0 : index
            %c10_35 = arith.constant 10 : index
            %c0_36 = arith.constant 0 : index
            %c1_37 = arith.constant 1 : index
            %34 = affine.apply #map2()
            %true_38 = arith.constant true
            %true_39 = arith.constant true
            %true_40 = arith.constant true
            %true_41 = arith.constant true
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %4 source_core(%24, %arg2) source_slice(%c9, %c0_10) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %5 source_core(%26, %arg2) source_slice(%c0_12, %c0_13) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %6 source_core(%arg1, %28) source_slice(%c0_16, %c0_17) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            tileflow.bulk_transfer %0 into %7 source_core(%arg1, %30) source_slice(%c0_20, %c9_21) epoch(%arg4) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
            task.epoch_wait %4 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %7 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %35 = arith.cmpi sge, %arg5, %c0_22 : index
                %36 = arith.cmpi slt, %arg5, %c1_23 : index
                %37 = arith.andi %35, %36 : i1
                %38 = arith.cmpi sge, %arg6, %c0_24 : index
                %39 = arith.cmpi slt, %arg6, %c10_25 : index
                %40 = arith.andi %38, %39 : i1
                %41 = arith.andi %37, %40 : i1
                %42 = affine.apply #map3(%arg6)
                %43 = tileflow.read_l1_buffer %4 at(%31, %42) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %44 = scf.if %41 -> (!d2mtile.tile<bf16>) {
                  scf.yield %43 : !d2mtile.tile<bf16>
                } else {
                  %99 = affine.apply #map4(%arg1, %arg2, %arg5, %arg6)
                  %100 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %101 = tileflow.read_l1_buffer %0 at(%99, %100) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %101 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %45 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %46 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %47 = tileflow.read_l1_buffer %0 at(%45, %46) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %48 = tileflow.roll %47 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %49 = tileflow.lane_copy %48 with %47 if(%true_38) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %50 = tileflow.lane_copy %49 with %44 if(%19) axis row src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %8 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) to %50 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %8 epoch(%arg4) : !tileplan.l1_buffer
                %51 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %52 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %53 = tileflow.read_l1_buffer %0 at(%51, %52) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %54 = arith.cmpi sge, %arg5, %c9_26 : index
                %55 = arith.cmpi slt, %arg5, %c10_27 : index
                %56 = arith.andi %54, %55 : i1
                %57 = arith.cmpi sge, %arg6, %c0_28 : index
                %58 = arith.cmpi slt, %arg6, %c10_29 : index
                %59 = arith.andi %57, %58 : i1
                %60 = arith.andi %56, %59 : i1
                %61 = affine.apply #map3(%arg6)
                %62 = tileflow.read_l1_buffer %5 at(%32, %61) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %63 = scf.if %60 -> (!d2mtile.tile<bf16>) {
                  scf.yield %62 : !d2mtile.tile<bf16>
                } else {
                  %99 = affine.apply #map7(%arg1, %arg2, %arg5, %arg6)
                  %100 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                  %101 = tileflow.read_l1_buffer %0 at(%99, %100) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %101 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                %64 = tileflow.roll %53 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %65 = tileflow.lane_copy %64 with %53 if(%true_39) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                %66 = tileflow.lane_copy %65 with %63 if(%20) axis row src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %9 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) to %66 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %9 epoch(%arg4) : !tileplan.l1_buffer
                %67 = arith.cmpi sge, %arg5, %c0_30 : index
                %68 = arith.cmpi slt, %arg5, %c10_31 : index
                %69 = arith.andi %67, %68 : i1
                %70 = arith.cmpi sge, %arg6, %c9_32 : index
                %71 = arith.cmpi slt, %arg6, %c10_33 : index
                %72 = arith.andi %70, %71 : i1
                %73 = arith.andi %69, %72 : i1
                %74 = affine.apply #map3(%arg5)
                %75 = tileflow.read_l1_buffer %6 at(%74, %33) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %76 = scf.if %73 -> (!d2mtile.tile<bf16>) {
                  scf.yield %75 : !d2mtile.tile<bf16>
                } else {
                  %99 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %100 = affine.apply #map8(%arg1, %arg2, %arg5, %arg6)
                  %101 = tileflow.read_l1_buffer %0 at(%99, %100) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %101 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                task.epoch_wait %10 epoch(%arg4) : !tileplan.l1_buffer
                %77 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %78 = tileflow.roll %77 axis row by 1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %79 = tileflow.read_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %80 = tileflow.lane_copy %78 with %79 if(%true_40) axis row src_lane 16 dst_lane 15 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %11 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) to %80 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %11 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %10 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %12 epoch(%arg4) : !tileplan.l1_buffer
                %81 = tileflow.read_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %82 = tileflow.lane_copy %81 with %76 if(%21) axis col src_lane 0 dst_lane 31 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %13 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) to %82 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %13 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %12 epoch(%arg4) : !tileplan.l1_buffer
                %83 = arith.cmpi sge, %arg5, %c0_34 : index
                %84 = arith.cmpi slt, %arg5, %c10_35 : index
                %85 = arith.andi %83, %84 : i1
                %86 = arith.cmpi sge, %arg6, %c0_36 : index
                %87 = arith.cmpi slt, %arg6, %c1_37 : index
                %88 = arith.andi %86, %87 : i1
                %89 = arith.andi %85, %88 : i1
                %90 = affine.apply #map3(%arg5)
                %91 = tileflow.read_l1_buffer %7 at(%90, %34) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %92 = scf.if %89 -> (!d2mtile.tile<bf16>) {
                  scf.yield %91 : !d2mtile.tile<bf16>
                } else {
                  %99 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                  %100 = affine.apply #map9(%arg1, %arg2, %arg5, %arg6)
                  %101 = tileflow.read_l1_buffer %0 at(%99, %100) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                  scf.yield %101 : !d2mtile.tile<bf16>
                } {task.engine_label = "dm"}
                task.epoch_wait %14 epoch(%arg4) : !tileplan.l1_buffer
                %93 = tileflow.read_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %94 = tileflow.roll %93 axis row by -1 {task.engine_label = "dm", tileflow.execution = "dm"} : !d2mtile.tile<bf16>
                %95 = tileflow.read_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %96 = tileflow.lane_copy %94 with %95 if(%true_41) axis row src_lane 15 dst_lane 16 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %15 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %15 at(%arg5, %arg6) epoch(%arg4) to %96 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %15 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %14 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %16 epoch(%arg4) : !tileplan.l1_buffer
                %97 = tileflow.read_l1_buffer %16 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %98 = tileflow.lane_copy %97 with %92 if(%22) axis col src_lane 31 dst_lane 0 {task.engine_label = "dm", tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
                task.epoch_acquire %17 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %17 at(%arg5, %arg6) epoch(%arg4) to %98 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %17 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_free %16 epoch(%arg4) : !tileplan.l1_buffer
              }
            }
            task.epoch_free %7 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %6 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %5 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %4 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @compute_0 engine(<compute>) domain(<phase>) reads(%4 : !tileplan.l1_buffer [full], %5 : !tileplan.l1_buffer [full], %6 : !tileplan.l1_buffer [full], %7 : !tileplan.l1_buffer [full], %2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) writes() after [#task.dep<@dm0_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            %c0 = arith.constant 0 : index
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %c0_5 = arith.constant 0 : index
            task.epoch_acquire %18 epoch(%arg4) : !tileplan.l1_buffer
            %19 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
            %20 = d2mtile.fill "2.000000e-01" -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
            %c0_6 = arith.constant 0 : index
            %c0_7 = arith.constant 0 : index
            tileflow.write_l1_buffer %18 at(%c0_6, %c0_7) epoch(%arg4) to %20 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
            task.epoch_ready %18 epoch(%arg4) : !tileplan.l1_buffer
            d2mtile.dst_release %19 : !d2mtile.dst_bank
            task.epoch_wait %2 : !tileplan.l1_buffer
            task.epoch_wait %3 : !tileplan.l1_buffer
            task.epoch_wait %0 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_wait %18 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_acquire %1 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %21 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %22 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                task.epoch_acquire %10 epoch(%arg4) : !tileplan.l1_buffer
                %23 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %24 = tileflow.read_l1_buffer %0 at(%21, %22) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %25 = d2mtile.unary tile_transpose %24 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %10 at(%arg5, %arg6) epoch(%arg4) to %25 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %10 epoch(%arg4) : !tileplan.l1_buffer
                d2mtile.dst_release %23 : !d2mtile.dst_bank
                task.epoch_acquire %12 epoch(%arg4) : !tileplan.l1_buffer
                %26 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                task.epoch_wait %11 epoch(%arg4) : !tileplan.l1_buffer
                %27 = tileflow.read_l1_buffer %11 at(%arg5, %arg6) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %28 = d2mtile.unary tile_transpose %27 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %11 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %12 at(%arg5, %arg6) epoch(%arg4) to %28 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %12 epoch(%arg4) : !tileplan.l1_buffer
                d2mtile.dst_release %26 : !d2mtile.dst_bank
                task.epoch_acquire %14 epoch(%arg4) : !tileplan.l1_buffer
                %29 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %30 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %31 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %32 = tileflow.read_l1_buffer %0 at(%30, %31) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %33 = d2mtile.unary tile_transpose %32 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %14 at(%arg5, %arg6) epoch(%arg4) to %33 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %14 epoch(%arg4) : !tileplan.l1_buffer
                d2mtile.dst_release %29 : !d2mtile.dst_bank
                task.epoch_acquire %16 epoch(%arg4) : !tileplan.l1_buffer
                %34 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                task.epoch_wait %15 epoch(%arg4) : !tileplan.l1_buffer
                %35 = tileflow.read_l1_buffer %15 at(%arg5, %arg6) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %36 = d2mtile.unary tile_transpose %35 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 0 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %15 epoch(%arg4) : !tileplan.l1_buffer
                tileflow.write_l1_buffer %16 at(%arg5, %arg6) epoch(%arg4) to %36 : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                task.epoch_ready %16 epoch(%arg4) : !tileplan.l1_buffer
                d2mtile.dst_release %34 : !d2mtile.dst_bank
                %37 = d2mtile.dst_acquire capacity 8 : !d2mtile.dst_bank
                %38 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %39 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %40 = tileflow.read_l1_buffer %0 at(%38, %39) epoch(%arg4) {tt.dst_slot = 0 : i64} : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                task.epoch_wait %17 epoch(%arg4) : !tileplan.l1_buffer
                %41 = tileflow.read_l1_buffer %17 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %42 = d2mtile.binary tile_add %40, %41 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %17 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %13 epoch(%arg4) : !tileplan.l1_buffer
                %43 = tileflow.read_l1_buffer %13 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %44 = d2mtile.binary tile_add %42, %43 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %13 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %9 epoch(%arg4) : !tileplan.l1_buffer
                %45 = tileflow.read_l1_buffer %9 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %46 = d2mtile.binary tile_add %44, %45 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %9 epoch(%arg4) : !tileplan.l1_buffer
                task.epoch_wait %8 epoch(%arg4) : !tileplan.l1_buffer
                %47 = tileflow.read_l1_buffer %8 at(%arg5, %arg6) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %48 = d2mtile.binary tile_add %46, %47 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                task.epoch_free %8 epoch(%arg4) : !tileplan.l1_buffer
                %c0_8 = arith.constant 0 : index
                %c0_9 = arith.constant 0 : index
                %49 = tileflow.read_l1_buffer %18 at(%c0_8, %c0_9) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %50 = d2mtile.binary tile_mul %48, %49 -> "source_ssa" {task.engine_label = "compute", tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                %c9 = arith.constant 9 : index
                %51 = arith.cmpi eq, %arg5, %c9 : index
                %c1 = arith.constant 1 : index
                %c0_10 = arith.constant 0 : index
                %52 = arith.select %51, %c1, %c0_10 : index
                %c0_11 = arith.constant 0 : index
                %53 = tileflow.read_l1_buffer %2 at(%52, %c0_11) {tt.dst_slot = 2 : i64} : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %54 = affine.apply #map(%arg1, %arg5)
                %55 = affine.apply #map10()[%54]
                %56 = arith.cmpi slt, %55, %c0 : index
                %57 = affine.apply #map1(%arg1, %arg5)
                %58 = affine.apply #map11()[%57]
                %59 = arith.cmpi slt, %58, %c0_3 : index
                %60 = arith.ori %56, %59 : i1
                %61 = tileflow.compute_select %50 with %40 active(%53 : !tileflow.mask<bf16>) if(%60) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>, tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                %c0_12 = arith.constant 0 : index
                %c9_13 = arith.constant 9 : index
                %62 = arith.cmpi eq, %arg6, %c9_13 : index
                %c1_14 = arith.constant 1 : index
                %c0_15 = arith.constant 0 : index
                %63 = arith.select %62, %c1_14, %c0_15 : index
                %64 = tileflow.read_l1_buffer %3 at(%c0_12, %63) {tt.dst_slot = 2 : i64} : !tileplan.l1_buffer -> !tileflow.mask<bf16>
                %65 = affine.apply #map(%arg2, %arg6)
                %66 = affine.apply #map10()[%65]
                %67 = arith.cmpi slt, %66, %c0_4 : index
                %68 = affine.apply #map1(%arg2, %arg6)
                %69 = affine.apply #map11()[%68]
                %70 = arith.cmpi slt, %69, %c0_5 : index
                %71 = arith.ori %67, %70 : i1
                %72 = tileflow.compute_select %61 with %40 active(%64 : !tileflow.mask<bf16>) if(%71) {task.engine_label = "compute", tileflow.overlay = #tileflow.overlay_realization<compute_select>, tt.dst_slot = 1 : i64} : !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %1 at(%arg5, %arg6) epoch(%arg4) to %72 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
                d2mtile.dst_release %37 : !d2mtile.dst_bank
              }
            }
            task.epoch_ready %1 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %18 epoch(%arg4) : !tileplan.l1_buffer
            task.epoch_free %0 epoch(%arg4) : !tileplan.l1_buffer
          }
          task.group @dm0_5 engine(<dm0>) domain(<phase>) reads() writes() after [#task.dep<@compute_0, raw>] {
            %c10_1 = arith.constant 10 : index
            %c10_2 = arith.constant 10 : index
            task.epoch_acquire %0 epoch(%arg4) {task.prime_before_loop} : !tileplan.l1_buffer
            task.epoch_wait %1 epoch(%arg4) : !tileplan.l1_buffer
            tileplan.shard_for (%arg5) in (%c10_1) axis <row> {
              tileplan.shard_for (%arg6) in (%c10_2) axis <col> {
                %19 = affine.apply #map6(%arg1, %arg2, %arg5, %arg6)
                %20 = affine.apply #map5(%arg1, %arg2, %arg5, %arg6)
                %21 = tileflow.read_l1_buffer %1 at(%19, %20) epoch(%arg4) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                tileflow.write_l1_buffer %0 at(%arg5, %arg6) epoch(%arg4) to %21 {task.engine_label = "dm"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
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

