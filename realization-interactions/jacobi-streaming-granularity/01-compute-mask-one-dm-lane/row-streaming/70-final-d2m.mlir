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
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xbf16>) -> memref<3200x3200xbf16> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 870400 : i64, tt.l1_persistent_base = 460800 : i64, tt.l1_pool = 460800 : i64} {
    %alloc = memref.alloc() {address = 571904 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 776704 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 315904 : i64, alignment = 16 : i64} : memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 320000 : i64, alignment = 16 : i64} : memref<1x2x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<4096x2048, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 324096 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 344576 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_6 = memref.alloc() {address = 365056 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_7 = memref.alloc() {address = 385536 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_8 = memref.alloc() {address = 406016 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_9 = memref.alloc() {address = 446976 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_10 = memref.alloc() {address = 487936 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_11 = memref.alloc() {address = 528896 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_12 = memref.alloc() {address = 569856 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_13 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_14 = memref.alloc() {address = 28868608 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>
    d2m.to_device %arg0, %alloc_13 layout = #layout : memref<3200x3200xbf16> into memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_15 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_16 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_13 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        outs(%alloc_14 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        additionalArgs(%alloc_15, %alloc_16 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
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
        d2m.remote_load %alloc_13[%core0, %core1] into %2 srcSlice[%4, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into !d2m.cb<memref<32x320xbf16, #l1>>
        d2m.remote_store %alloc_14[%core0, %core1] from %3 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
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
        ins(%alloc_14 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_load %alloc_14[%core0, %core1] into %2 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
    ^compute0:
    }
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc, %alloc_0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>, memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        additionalArgs(%alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %alloc_6, %alloc_7, %alloc_8, %alloc_9, %alloc_10, %alloc_11, %alloc_12, %0, %1 : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<2x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x2x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<4096x2048, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c20480 = arith.constant 20480 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c9 = arith.constant 9 : index
      %c-1 = arith.constant -1 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c15 = arith.constant 15 : index
      %c16 = arith.constant 16 : index
      %c31 = arith.constant 31 : index
      %c1000 = arith.constant 1000 : index
      %2 = d2m.get_cb(3) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(4) resolution_stage =  compile : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
      %4 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.get_cb(7) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %7 = d2m.get_cb(8) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
      %8 = d2m.get_cb(9) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
      %9 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %10 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %11 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %12 = d2m.get_cb(13) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %13 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %14 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %15 = d2m.reserve %3 {pin_cb = 4 : i64} : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
      %16 = affine.apply #map()[%core0]
      %17 = arith.subi %c0, %16 : index
      %18 = arith.maxsi %17, %c0 : index
      %19 = arith.shli %c1, %18 : index
      %20 = arith.subi %19, %c1 : index
      d2m.write_row_mask_tile(%20) to %3 at %c0 of %c2 : index, !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index
      %21 = affine.apply #map1()[%core0]
      %22 = arith.addi %21, %c1 : index
      %23 = arith.maxsi %22, %c0 : index
      %24 = arith.shli %c1, %23 : index
      %25 = arith.subi %24, %c1 : index
      d2m.write_row_mask_tile(%25) to %3 at %c1 of %c2 : index, !d2m.cb<memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index
      d2m.push %3 {pin_cb = 4 : i64} : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
      %26 = d2m.reserve %4 {pin_cb = 5 : i64} : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x2x!ttcore.tile<32x32, bf16>, #l1>
      %27 = affine.apply #map()[%core1]
      %28 = arith.subi %c0, %27 : index
      %29 = arith.maxsi %28, %c0 : index
      %30 = arith.shli %c1, %29 : index
      %31 = arith.subi %30, %c1 : index
      d2m.write_col_mask_tile(%31) to %4 at %c0 of %c2 : index, !d2m.cb<memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>, index, index
      %32 = affine.apply #map1()[%core1]
      %33 = arith.addi %32, %c1 : index
      %34 = arith.maxsi %33, %c0 : index
      %35 = arith.shli %c1, %34 : index
      %36 = arith.subi %35, %c1 : index
      d2m.write_col_mask_tile(%36) to %4 at %c1 of %c2 : index, !d2m.cb<memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>, index, index
      d2m.push %4 {pin_cb = 5 : i64} : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>
      %37 = d2m.reserve %14 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %14 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %38 = arith.addi %core0, %c-1 : index
        %39 = arith.maxsi %38, %c0 : index
        %40 = arith.addi %core0, %c1 : index
        %41 = arith.minsi %40, %c9 : index
        %42 = arith.addi %core1, %c1 : index
        %43 = arith.minsi %42, %c9 : index
        %44 = arith.addi %core1, %c-1 : index
        %45 = arith.maxsi %44, %c0 : index
        %46 = d2m.wait %14 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %47 = d2m.wait %13 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %48 = d2m.get_arg(15) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %48, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %48, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc[%39, %core1] into %5 srcSlice[%c9, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc[%41, %core1] into %6 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc_0[%core0, %43] into %7 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc_0[%core0, %45] into %8 srcSlice[%c0, %c9] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        %49 = d2m.wait %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %50 = d2m.wait %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %51 = d2m.wait %7 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x1x!ttcore.tile<32x32, bf16>, #l1>
        %52 = d2m.wait %8 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x1x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %56 = d2m.reserve %9 {pin_cb = 10 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %57 = d2m.reserve %10 {pin_cb = 11 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %58 = d2m.reserve %11 {pin_cb = 12 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %59 = d2m.reserve %12 {pin_cb = 13 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %60 = affine.apply #map2(%arg2)
          %61 = arith.muli %60, %c20480 : index
          %tx_20 = d2m.dma_read %46[%61], %56[%c0], <10> {srcByteOffset = -32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx_20 : !d2m.mem_tx<read>
          %tx_21 = d2m.dma_read %46[%61], %57[%c0], <10> {srcByteOffset = 32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx_21 : !d2m.mem_tx<read>
          %tx_22 = d2m.dma_read %47[%61], %58[%c0], <10> {srcByteOffset = 32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx_22 : !d2m.mem_tx<read>
          %tx_23 = d2m.dma_read %47[%61], %59[%c0], <10> {srcByteOffset = -32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx_23 : !d2m.mem_tx<read>
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %62 = arith.cmpi slt, %arg2, %c1 : index
            %63 = affine.apply #map2(%arg3)
            %64 = arith.muli %60, %c10 : index
            %65 = arith.addi %64, %63 : index
            d2m.copy_single_row_tile(%46, %c15) to %56 at %c16 from %65, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %62 {
              d2m.copy_single_row_tile(%49, %c31) to %56 at %c0 from %63, %arg3 : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %69 = affine.apply #map3(%arg2)[%core0]
              %70 = arith.muli %69, %c10 : index
              %71 = arith.addi %70, %63 : index
              d2m.copy_single_row_tile(%46, %c31) to %56 at %c0 from %71, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            %66 = arith.cmpi sge, %arg2, %c9 : index
            d2m.copy_single_row_tile(%46, %c16) to %57 at %c15 from %65, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %66 {
              d2m.copy_single_row_tile(%50, %c0) to %57 at %c31 from %63, %arg3 : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %69 = affine.apply #map4(%arg2)[%core0]
              %70 = arith.muli %69, %c10 : index
              %71 = arith.addi %70, %63 : index
              d2m.copy_single_row_tile(%46, %c0) to %57 at %c31 from %71, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            %67 = arith.cmpi sge, %arg3, %c9 : index
            d2m.copy_single_row_tile(%47, %c16) to %58 at %c15 from %65, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %67 {
              d2m.copy_single_row_tile(%51, %c0) to %58 at %c31 from %60, %arg3 : memref<10x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %69 = affine.apply #map4(%arg3)[%core1]
              %70 = arith.addi %64, %69 : index
              d2m.copy_single_row_tile(%47, %c0) to %58 at %c31 from %70, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            %68 = arith.cmpi slt, %arg3, %c1 : index
            d2m.copy_single_row_tile(%47, %c15) to %59 at %c16 from %65, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %68 {
              d2m.copy_single_row_tile(%52, %c31) to %59 at %c0 from %60, %arg3 : memref<10x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %69 = affine.apply #map3(%arg3)[%core1]
              %70 = arith.addi %64, %69 : index
              d2m.copy_single_row_tile(%47, %c31) to %59 at %c0 from %70, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
          }
          d2m.push %12 {pin_cb = 13 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.push %11 {pin_cb = 12 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.push %10 {pin_cb = 11 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.push %9 {pin_cb = 10 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        }
        d2m.pop %8 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %7 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        %53 = d2m.get_arg(16) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %53, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %53, %c100 reset %c0 : !d2m.local_semaphore
        d2m.pop %13 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        %54 = d2m.reserve %14 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %55 = d2m.wait %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %tx = d2m.dma_read %55[%c0], %54[%c0], <100> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx : !d2m.mem_tx<read>
        d2m.pop %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.push %14 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
      d2m.atomic_barrier
    }, {
    ^datamovement1:
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 2.001950e-01 : bf16
      %c2 = arith.constant 2 : index
      %c9 = arith.constant 9 : index
      %c1000 = arith.constant 1000 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(3) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(4) resolution_stage =  compile : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>>
      %4 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %7 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %8 = d2m.get_cb(13) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %9 = d2m.get_cb(14) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %10 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %11 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %12 = d2m.wait %11 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %13 = d2m.reserve %10 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %dst_20 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %21 = memref.load %12[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            memref.store %21, %dst_20[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %22 = memref.load %dst_20[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %23 = "d2m.tile_transpose"(%22) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %23, %dst_20[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %24 = memref.load %dst_20[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            memref.store %24, %13[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          }
        }
        d2m.push %10 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        %14 = d2m.reserve %9 {pin_cb = 14 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        %15 = d2m.tile_fill(%cst) : bf16 -> <32x32, bf16>
        memref.store %15, %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        %16 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        memref.store %16, %14[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.push %9 {pin_cb = 14 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        %17 = d2m.wait %3 : <memref<2x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
        %18 = d2m.wait %4 : <memref<1x2x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x2x!ttcore.tile<32x32, bf16>, #l1>
        %19 = d2m.wait %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        %20 = d2m.reserve %2 {pin_cb = 3 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %21 = d2m.wait %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %22 = d2m.wait %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %23 = d2m.wait %7 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %24 = d2m.wait %8 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %25 = affine.apply #map2(%arg2)
            %26 = affine.apply #map2(%arg3)
            %dst_20 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %27 = memref.load %12[%25, %26] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            memref.store %27, %dst_20[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %28 = memref.load %dst_20[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %29 = memref.load %23[%c0, %arg3] : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
            %30 = memref.load %24[%c0, %arg3] : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
            %31 = "d2m.tile_add"(%30, %29) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %31, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %32 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %33 = "d2m.tile_transpose"(%32) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %33, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %34 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %35 = memref.load %22[%c0, %arg3] : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
            %36 = "d2m.tile_add"(%28, %35) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %36, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %37 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %38 = memref.load %21[%c0, %arg3] : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
            %39 = "d2m.tile_add"(%37, %38) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %39, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %40 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %41 = "d2m.tile_add"(%40, %34) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %41, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %42 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %43 = memref.load %19[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %44 = "d2m.tile_mul"(%42, %43) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %44, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %45 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %46 = arith.cmpi eq, %arg2, %c9 : index
            %47 = arith.select %46, %c1, %c0 : index
            %48 = affine.apply #map5(%arg2)[%core0]
            %49 = arith.cmpi slt, %48, %c0 : index
            %50 = affine.apply #map6(%arg2)[%core0]
            %51 = arith.cmpi slt, %50, %c0 : index
            %52 = arith.ori %49, %51 : i1
            %53 = scf.if %52 -> (!ttcore.tile<32x32, bf16>) {
              %62 = arith.cmpi eq, %47, %c1 : index
              %63 = memref.load %17[%47, %c0] : memref<2x1x!ttcore.tile<32x32, bf16>, #l1>
              memref.store %63, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
              %64 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
              %65 = scf.if %62 -> (!ttcore.tile<32x32, bf16>) {
                %66 = "d2m.tile_where"(%64, %45, %28) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
                memref.store %66, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                %67 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                scf.yield %67 : !ttcore.tile<32x32, bf16>
              } else {
                %66 = "d2m.tile_where"(%64, %28, %45) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
                memref.store %66, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                %67 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                scf.yield %67 : !ttcore.tile<32x32, bf16>
              }
              scf.yield %65 : !ttcore.tile<32x32, bf16>
            } else {
              scf.yield %45 : !ttcore.tile<32x32, bf16>
            }
            %54 = arith.cmpi eq, %arg3, %c9 : index
            %55 = arith.select %54, %c1, %c0 : index
            %56 = affine.apply #map5(%arg3)[%core1]
            %57 = arith.cmpi slt, %56, %c0 : index
            %58 = affine.apply #map6(%arg3)[%core1]
            %59 = arith.cmpi slt, %58, %c0 : index
            %60 = arith.ori %57, %59 : i1
            %61 = scf.if %60 -> (!ttcore.tile<32x32, bf16>) {
              %62 = arith.cmpi eq, %55, %c1 : index
              %63 = memref.load %18[%c0, %55] : memref<1x2x!ttcore.tile<32x32, bf16>, #l1>
              memref.store %63, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
              %64 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
              %65 = scf.if %62 -> (!ttcore.tile<32x32, bf16>) {
                %66 = "d2m.tile_where"(%64, %53, %28) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
                memref.store %66, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                %67 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                scf.yield %67 : !ttcore.tile<32x32, bf16>
              } else {
                %66 = "d2m.tile_where"(%64, %28, %53) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
                memref.store %66, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                %67 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
                scf.yield %67 : !ttcore.tile<32x32, bf16>
              }
              scf.yield %65 : !ttcore.tile<32x32, bf16>
            } else {
              scf.yield %53 : !ttcore.tile<32x32, bf16>
            }
            memref.store %61, %20[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          }
          d2m.pop %8 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.pop %7 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.pop %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.pop %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        }
        d2m.push %2 {pin_cb = 3 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %11 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
    }
    %alloc_17 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_18 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc_14 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_store %alloc_14[%core0, %core1] from %2 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.reserve %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_14 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc_13 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        additionalArgs(%alloc_17, %alloc_18 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
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
        d2m.remote_load %alloc_14[%core0, %core1] into %3 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_store %alloc_13[%core0, %core1] from %2 srcSlice[%4, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> from !d2m.cb<memref<32x320xbf16, #l1>>
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
    %alloc_19 = memref.alloc() : memref<3200x3200xbf16>
    d2m.to_host %alloc_13, %alloc_19 layout = #layout : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into memref<3200x3200xbf16>
    return %alloc_19 : memref<3200x3200xbf16>
  }
}
