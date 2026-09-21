#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 3200x3200, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<()[s0] -> (s0 * 320 - 1)>
#map1 = affine_map<()[s0] -> (s0 * -320 + 2910)>
#map2 = affine_map<(d0) -> (d0 mod 10)>
#map3 = affine_map<(d0)[s0] -> (d0 + ((d0 + s0 * 10 - 1) floordiv 10) * -10 + s0 * 10 - 1)>
#map4 = affine_map<(d0)[s0] -> (d0 + ((d0 + s0 * 10 + 1) floordiv 10) * -10 + s0 * 10 + 1)>
#map5 = affine_map<(d0)[s0] -> (d0 * 32 + s0 * 320 - 1)>
#map6 = affine_map<(d0)[s0] -> (d0 * -32 - s0 * 320 + 3167)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xbf16>) -> memref<3200x3200xbf16> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 542720 : i64, tt.l1_persistent_base = 337920 : i64, tt.l1_pool = 337920 : i64} {
    %alloc = memref.alloc() {address = 449024 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 315904 : i64, alignment = 16 : i64} : memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 320000 : i64, alignment = 16 : i64} : memref<1x2x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<4096x2048, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 324096 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 344576 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 365056 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_6 = memref.alloc() {address = 385536 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_7 = memref.alloc() {address = 406016 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_8 = memref.alloc() {address = 410112 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_9 = memref.alloc() {address = 414208 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_10 = memref.alloc() {address = 418304 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_11 = memref.alloc() {address = 422400 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_12 = memref.alloc() {address = 426496 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_13 = memref.alloc() {address = 430592 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_14 = memref.alloc() {address = 434688 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_15 = memref.alloc() {address = 438784 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_16 = memref.alloc() {address = 442880 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_17 = memref.alloc() {address = 446976 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_18 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_19 = memref.alloc() {address = 28868608 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>
    d2m.to_device %arg0, %alloc_18 layout = #layout : memref<3200x3200xbf16> into memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_20 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_21 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_18 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        outs(%alloc_19 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        additionalArgs(%alloc_20, %alloc_21 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %4 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_18[%core0, %core1] into %2 srcSlice[%4, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into !d2m.cb<memref<32x320xbf16, #l1>>
        d2m.remote_store %alloc_19[%core0, %core1] from %3 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %4 = d2m.wait %2 : <memref<32x320xbf16, #l1>> -> memref<32x320xbf16, #l1>
        %5 = d2m.reserve %3 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %6 = "d2m.tile_tilize_block"(%4, %5) : (memref<32x320xbf16, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        d2m.pop %2 : <memref<32x320xbf16, #l1>>
        d2m.push %3 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_19 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_load %alloc_19[%core0, %core1] into %2 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
    ^compute0:
    }
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        additionalArgs(%alloc_0, %alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %alloc_6, %alloc_7, %alloc_8, %alloc_9, %alloc_10, %alloc_11, %alloc_12, %alloc_13, %alloc_14, %alloc_15, %alloc_16, %alloc_17, %0, %1 : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x2x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<4096x2048, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c9 = arith.constant 9 : index
      %c-1 = arith.constant -1 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c2048 = arith.constant 2048 : index
      %c15 = arith.constant 15 : index
      %c16 = arith.constant 16 : index
      %c31 = arith.constant 31 : index
      %c1000 = arith.constant 1000 : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
      %4 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %7 = d2m.get_cb(7) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
      %8 = d2m.get_cb(8) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
      %9 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %10 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %11 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %12 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %13 = d2m.get_cb(13) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %14 = d2m.get_cb(14) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %15 = d2m.get_cb(15) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %16 = d2m.get_cb(16) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %17 = d2m.get_cb(17) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %18 = d2m.get_cb(18) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %19 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %20 = d2m.reserve %3 {pin_cb = 3 : i64} : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      %21 = affine.apply #map()[%core0]
      %22 = arith.subi %c0, %21 : index
      %23 = arith.maxsi %22, %c0 : index
      %24 = arith.shli %c1, %23 : index
      %25 = arith.subi %24, %c1 : index
      d2m.write_row_mask_tile(%25) to %3 at %c0 of %c2 : index, !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index
      %26 = affine.apply #map1()[%core0]
      %27 = arith.addi %26, %c1 : index
      %28 = arith.maxsi %27, %c0 : index
      %29 = arith.shli %c1, %28 : index
      %30 = arith.subi %29, %c1 : index
      d2m.write_row_mask_tile(%30) to %3 at %c1 of %c2 : index, !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index
      d2m.push %3 {pin_cb = 3 : i64} : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
      %31 = d2m.reserve %4 {pin_cb = 4 : i64} : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x2x!ttcore.tile<32x32, bf16>, #l1>
      %32 = affine.apply #map()[%core1]
      %33 = arith.subi %c0, %32 : index
      %34 = arith.maxsi %33, %c0 : index
      %35 = arith.shli %c1, %34 : index
      %36 = arith.subi %35, %c1 : index
      d2m.write_col_mask_tile(%36) to %4 at %c0 of %c2 : index, !d2m.cb<memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>, index, index
      %37 = affine.apply #map1()[%core1]
      %38 = arith.addi %37, %c1 : index
      %39 = arith.maxsi %38, %c0 : index
      %40 = arith.shli %c1, %39 : index
      %41 = arith.subi %40, %c1 : index
      d2m.write_col_mask_tile(%41) to %4 at %c1 of %c2 : index, !d2m.cb<memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>, index, index
      d2m.push %4 {pin_cb = 4 : i64} : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>
      %42 = d2m.reserve %19 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %19 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %43 = arith.addi %core0, %c-1 : index
        %44 = arith.maxsi %43, %c0 : index
        %45 = arith.addi %core0, %c1 : index
        %46 = arith.minsi %45, %c9 : index
        %47 = arith.addi %core1, %c1 : index
        %48 = arith.minsi %47, %c9 : index
        %49 = arith.addi %core1, %c-1 : index
        %50 = arith.maxsi %49, %c0 : index
        %51 = d2m.wait %19 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %52 = d2m.get_arg(20) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %52, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %52, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc[%44, %core1] into %5 srcSlice[%c9, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc[%46, %core1] into %6 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc[%core0, %48] into %7 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc[%core0, %50] into %8 srcSlice[%c0, %c9] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        %53 = d2m.wait %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %54 = d2m.wait %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %55 = d2m.wait %7 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x1x!ttcore.tile<32x32, bf16>, #l1>
        %56 = d2m.wait %8 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x1x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %60 = arith.cmpi slt, %arg2, %c1 : index
            %61 = affine.apply #map2(%arg3)
            %62 = affine.apply #map2(%arg2)
            %63 = d2m.reserve %9 {pin_cb = 9 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %64 = arith.muli %62, %c10 : index
            %65 = arith.addi %64, %61 : index
            %66 = arith.muli %65, %c2048 : index
            %tx_25 = d2m.dma_read %51[%66], %63[%c0], <1> {srcByteOffset = -32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_25 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%51, %c15) to %63 at %c16 from %65, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %60 {
              d2m.copy_single_row_tile(%53, %c31) to %63 at %c0 from %61, %c0 : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %79 = affine.apply #map3(%arg2)[%core0]
              %80 = arith.muli %79, %c10 : index
              %81 = arith.addi %80, %61 : index
              d2m.copy_single_row_tile(%51, %c31) to %63 at %c0 from %81, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            d2m.push %9 {pin_cb = 9 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %67 = arith.cmpi sge, %arg2, %c9 : index
            %68 = d2m.reserve %10 {pin_cb = 10 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %tx_26 = d2m.dma_read %51[%66], %68[%c0], <1> {srcByteOffset = 32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_26 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%51, %c16) to %68 at %c15 from %65, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %67 {
              d2m.copy_single_row_tile(%54, %c0) to %68 at %c31 from %61, %c0 : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %79 = affine.apply #map4(%arg2)[%core0]
              %80 = arith.muli %79, %c10 : index
              %81 = arith.addi %80, %61 : index
              d2m.copy_single_row_tile(%51, %c0) to %68 at %c31 from %81, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            d2m.push %10 {pin_cb = 10 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %69 = arith.cmpi sge, %arg3, %c9 : index
            %70 = d2m.wait %11 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %71 = d2m.reserve %12 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %tx_27 = d2m.dma_read %70[%c0], %71[%c0], <1> {srcByteOffset = 32 : si32} : (memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_27 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%70, %c16) to %71 at %c15 from %c0, %c0 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            d2m.push %12 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            d2m.pop %11 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %72 = d2m.wait %13 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %73 = d2m.reserve %14 {pin_cb = 14 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %tx_28 = d2m.dma_read %72[%c0], %73[%c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_28 : !d2m.mem_tx<read>
            scf.if %69 {
              d2m.copy_single_col_tile(%55, %c0) to %73 at %c31 from %62, %c0 : memref<10x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %79 = affine.apply #map4(%arg3)[%core1]
              %80 = arith.addi %64, %79 : index
              d2m.copy_single_col_tile(%51, %c0) to %73 at %c31 from %80, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            d2m.push %14 {pin_cb = 14 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            d2m.pop %13 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %74 = arith.cmpi slt, %arg3, %c1 : index
            %75 = d2m.wait %15 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %76 = d2m.reserve %16 {pin_cb = 16 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %tx_29 = d2m.dma_read %75[%c0], %76[%c0], <1> {srcByteOffset = -32 : si32} : (memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_29 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%75, %c15) to %76 at %c16 from %c0, %c0 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            d2m.push %16 {pin_cb = 16 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            d2m.pop %15 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %77 = d2m.wait %17 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %78 = d2m.reserve %18 {pin_cb = 18 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %tx_30 = d2m.dma_read %77[%c0], %78[%c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_30 : !d2m.mem_tx<read>
            scf.if %74 {
              d2m.copy_single_col_tile(%56, %c31) to %78 at %c0 from %62, %c0 : memref<10x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %79 = affine.apply #map3(%arg3)[%core1]
              %80 = arith.addi %64, %79 : index
              d2m.copy_single_col_tile(%51, %c31) to %78 at %c0 from %80, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            d2m.push %18 {pin_cb = 18 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            d2m.pop %17 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
          }
        }
        d2m.pop %8 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %7 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        %57 = d2m.get_arg(21) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %57, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %57, %c100 reset %c0 : !d2m.local_semaphore
        %58 = d2m.reserve %19 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %59 = d2m.wait %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %tx = d2m.dma_read %59[%c0], %58[%c0], <100> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx : !d2m.mem_tx<read>
        d2m.pop %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.push %19 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
      d2m.atomic_barrier
    }, {
    ^datamovement1:
    }, {
      %cst = arith.constant 2.001950e-01 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9 = arith.constant 9 : index
      %c2 = arith.constant 2 : index
      %c10 = arith.constant 10 : index
      %c1000 = arith.constant 1000 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
      %4 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %7 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %8 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %9 = d2m.get_cb(13) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %10 = d2m.get_cb(14) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %11 = d2m.get_cb(15) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %12 = d2m.get_cb(16) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %13 = d2m.get_cb(17) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %14 = d2m.get_cb(18) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %15 = d2m.get_cb(19) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %16 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %17 = d2m.reserve %15 {pin_cb = 19 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        %18 = d2m.tile_fill(%cst) : bf16 -> <32x32, bf16>
        memref.store %18, %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        %19 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        memref.store %19, %17[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.push %15 {pin_cb = 19 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        %20 = d2m.wait %3 : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %21 = d2m.wait %4 : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x2x!ttcore.tile<32x32, bf16>, #l1>
        %22 = d2m.wait %16 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %23 = d2m.wait %15 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        %24 = d2m.reserve %2 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %25 = affine.apply #map2(%arg2)
            %26 = affine.apply #map2(%arg3)
            %27 = d2m.reserve %7 {pin_cb = 11 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %dst_25 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %28 = memref.load %22[%25, %26] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            memref.store %28, %dst_25[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %29 = memref.load %dst_25[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %30 = "d2m.tile_transpose"(%29) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %30, %dst_25[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %31 = memref.load %dst_25[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            memref.store %31, %27[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            d2m.push %7 {pin_cb = 11 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %32 = d2m.reserve %9 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %dst_26 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %33 = d2m.wait %8 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %34 = memref.load %33[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            memref.store %34, %dst_26[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %35 = memref.load %dst_26[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %36 = "d2m.tile_transpose"(%35) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %36, %dst_26[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %37 = memref.load %dst_26[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            d2m.pop %8 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            memref.store %37, %32[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            d2m.push %9 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %38 = d2m.reserve %11 {pin_cb = 15 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %dst_27 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %39 = memref.load %22[%25, %26] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            memref.store %39, %dst_27[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %40 = memref.load %dst_27[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %41 = "d2m.tile_transpose"(%40) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %41, %dst_27[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %42 = memref.load %dst_27[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            memref.store %42, %38[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            d2m.push %11 {pin_cb = 15 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %43 = d2m.reserve %13 {pin_cb = 17 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %dst_28 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %44 = d2m.wait %12 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %45 = memref.load %44[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            memref.store %45, %dst_28[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %46 = memref.load %dst_28[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %47 = "d2m.tile_transpose"(%46) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %47, %dst_28[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %48 = memref.load %dst_28[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            d2m.pop %12 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            memref.store %48, %43[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            d2m.push %13 {pin_cb = 17 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %dst_29 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %49 = memref.load %22[%25, %26] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            memref.store %49, %dst_29[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %50 = memref.load %dst_29[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %51 = d2m.wait %14 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %52 = memref.load %51[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %53 = "d2m.tile_add"(%50, %52) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %53, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %54 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            d2m.pop %14 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %55 = d2m.wait %10 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %56 = memref.load %55[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %57 = "d2m.tile_add"(%54, %56) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %57, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %58 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            d2m.pop %10 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %59 = d2m.wait %6 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %60 = memref.load %59[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %61 = "d2m.tile_add"(%58, %60) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %61, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %62 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            d2m.pop %6 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %63 = d2m.wait %5 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %64 = memref.load %63[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %65 = "d2m.tile_add"(%62, %64) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %65, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %66 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            d2m.pop %5 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %67 = memref.load %23[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %68 = "d2m.tile_mul"(%66, %67) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %68, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %69 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %70 = arith.cmpi eq, %arg2, %c9 : index
            %71 = arith.select %70, %c1, %c0 : index
            %72 = affine.apply #map5(%arg2)[%core0]
            %73 = arith.cmpi slt, %72, %c0 : index
            %74 = affine.apply #map6(%arg2)[%core0]
            %75 = arith.cmpi slt, %74, %c0 : index
            %76 = arith.ori %73, %75 : i1
            %77 = scf.if %76 -> (!ttcore.tile<32x32, bf16>) {
              %86 = arith.cmpi eq, %71, %c1 : index
              %87 = memref.load %20[%71, %c0] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
              memref.store %87, %dst_29[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
              %88 = memref.load %dst_29[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
              %89 = scf.if %86 -> (!ttcore.tile<32x32, bf16>) {
                %90 = "d2m.tile_where"(%88, %69, %50) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
                memref.store %90, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                %91 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                scf.yield %91 : !ttcore.tile<32x32, bf16>
              } else {
                %90 = "d2m.tile_where"(%88, %50, %69) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
                memref.store %90, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                %91 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                scf.yield %91 : !ttcore.tile<32x32, bf16>
              }
              scf.yield %89 : !ttcore.tile<32x32, bf16>
            } else {
              scf.yield %69 : !ttcore.tile<32x32, bf16>
            }
            %78 = arith.cmpi eq, %arg3, %c9 : index
            %79 = arith.select %78, %c1, %c0 : index
            %80 = affine.apply #map5(%arg3)[%core1]
            %81 = arith.cmpi slt, %80, %c0 : index
            %82 = affine.apply #map6(%arg3)[%core1]
            %83 = arith.cmpi slt, %82, %c0 : index
            %84 = arith.ori %81, %83 : i1
            %85 = scf.if %84 -> (!ttcore.tile<32x32, bf16>) {
              %86 = arith.cmpi eq, %79, %c1 : index
              %87 = memref.load %21[%c0, %79] : memref<1x2x!ttcore.tile<32x32, bf16>, #l1>
              memref.store %87, %dst_29[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
              %88 = memref.load %dst_29[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
              %89 = scf.if %86 -> (!ttcore.tile<32x32, bf16>) {
                %90 = "d2m.tile_where"(%88, %77, %50) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
                memref.store %90, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                %91 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                scf.yield %91 : !ttcore.tile<32x32, bf16>
              } else {
                %90 = "d2m.tile_where"(%88, %50, %77) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
                memref.store %90, %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                %91 = memref.load %dst_29[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                scf.yield %91 : !ttcore.tile<32x32, bf16>
              }
              scf.yield %89 : !ttcore.tile<32x32, bf16>
            } else {
              scf.yield %77 : !ttcore.tile<32x32, bf16>
            }
            memref.store %85, %24[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          }
        }
        d2m.push %2 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %15 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %16 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
    }
    %alloc_22 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_23 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc_19 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_store %alloc_19[%core0, %core1] from %2 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.reserve %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_19 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc_18 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        additionalArgs(%alloc_22, %alloc_23 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %4 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_19[%core0, %core1] into %3 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_store %alloc_18[%core0, %core1] from %2 srcSlice[%4, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> from !d2m.cb<memref<32x320xbf16, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %2 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %4 = d2m.wait %2 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %5 = d2m.reserve %3 : <memref<32x320xbf16, #l1>> -> memref<32x320xbf16, #l1>
        %6 = "d2m.tile_untilize_block"(%4, %5) : (memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, memref<32x320xbf16, #l1>) -> memref<32x320xbf16, #l1>
        d2m.pop %2 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.push %3 : <memref<32x320xbf16, #l1>>
      }
    }
    %alloc_24 = memref.alloc() : memref<3200x3200xbf16>
    d2m.to_host %alloc_18, %alloc_24 layout = #layout : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into memref<3200x3200xbf16>
    return %alloc_24 : memref<3200x3200xbf16>
  }
}
