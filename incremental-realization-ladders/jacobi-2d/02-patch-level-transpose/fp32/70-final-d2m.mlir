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
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xf32>) -> memref<3200x3200xf32> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 1069056 : i64, tt.l1_persistent_base = 659456 : i64, tt.l1_pool = 659456 : i64} {
    %alloc = memref.alloc() {address = 770560 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<10x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 520704 : i64, alignment = 16 : i64} : memref<2x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 528896 : i64, alignment = 16 : i64} : memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<8192x4096, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 537088 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 578048 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 619008 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_6 = memref.alloc() {address = 659968 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_7 = memref.alloc() {address = 700928 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_8 = memref.alloc() {address = 709120 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_9 = memref.alloc() {address = 717312 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_10 = memref.alloc() {address = 725504 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_11 = memref.alloc() {address = 733696 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_12 = memref.alloc() {address = 741888 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_13 = memref.alloc() {address = 750080 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_14 = memref.alloc() {address = 758272 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_15 = memref.alloc() {address = 766464 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_16 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram>
    %alloc_17 = memref.alloc() {address = 49348608 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>
    d2m.to_device %arg0, %alloc_16 layout = #layout : memref<3200x3200xf32> into memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram>
    %alloc_18 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xf32, #ttcore.cb_layout<1280x4, 1>, #l1>
    %alloc_19 = memref.alloc() {address = 152064 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_16 : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram>)
        outs(%alloc_17 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>)
        additionalArgs(%alloc_18, %alloc_19 : memref<32x320xf32, #ttcore.cb_layout<1280x4, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xf32, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %4 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_16[%core0, %core1] into %2 srcSlice[%4, %c0] : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram> into !d2m.cb<memref<32x320xf32, #l1>>
        d2m.remote_store %alloc_17[%core0, %core1] from %3 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram> from !d2m.cb<memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xf32, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %4 = d2m.wait %2 : <memref<32x320xf32, #l1>> -> memref<32x320xf32, #l1>
        %5 = d2m.reserve %3 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        %6 = "d2m.tile_tilize_block"(%4, %5) : (memref<32x320xf32, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #l1>) -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        d2m.pop %2 : <memref<32x320xf32, #l1>>
        d2m.push %3 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      }
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_17 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      d2m.remote_load %alloc_17[%core0, %core1] into %2 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0:
    }
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>)
        additionalArgs(%alloc_0, %alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %alloc_6, %alloc_7, %alloc_8, %alloc_9, %alloc_10, %alloc_11, %alloc_12, %alloc_13, %alloc_14, %alloc_15, %0, %1 : memref<10x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>, memref<2x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<8192x4096, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c9 = arith.constant 9 : index
      %c-1 = arith.constant -1 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c4096 = arith.constant 4096 : index
      %c15 = arith.constant 15 : index
      %c16 = arith.constant 16 : index
      %c31 = arith.constant 31 : index
      %c1000 = arith.constant 1000 : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<2x1x!ttcore.tile<32x32, f32>, #l1>>
      %4 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      %6 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      %7 = d2m.get_cb(7) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
      %8 = d2m.get_cb(8) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
      %9 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %10 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %11 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %12 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %13 = d2m.get_cb(13) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %14 = d2m.get_cb(14) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %15 = d2m.get_cb(15) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %16 = d2m.get_cb(16) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %17 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %18 = d2m.reserve %3 {pin_cb = 3 : i64} : <memref<2x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x1x!ttcore.tile<32x32, f32>, #l1>
      %19 = affine.apply #map()[%core0]
      %20 = arith.subi %c0, %19 : index
      %21 = arith.maxsi %20, %c0 : index
      %22 = arith.shli %c1, %21 : index
      %23 = arith.subi %22, %c1 : index
      d2m.write_row_mask_tile(%23) to %3 at %c0 of %c2 : index, !d2m.cb<memref<2x1x!ttcore.tile<32x32, f32>, #l1>>, index, index
      %24 = affine.apply #map1()[%core0]
      %25 = arith.addi %24, %c1 : index
      %26 = arith.maxsi %25, %c0 : index
      %27 = arith.shli %c1, %26 : index
      %28 = arith.subi %27, %c1 : index
      d2m.write_row_mask_tile(%28) to %3 at %c1 of %c2 : index, !d2m.cb<memref<2x1x!ttcore.tile<32x32, f32>, #l1>>, index, index
      d2m.push %3 {pin_cb = 3 : i64} : <memref<2x1x!ttcore.tile<32x32, f32>, #l1>>
      %29 = d2m.reserve %4 {pin_cb = 4 : i64} : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x2x!ttcore.tile<32x32, f32>, #l1>
      %30 = affine.apply #map()[%core1]
      %31 = arith.subi %c0, %30 : index
      %32 = arith.maxsi %31, %c0 : index
      %33 = arith.shli %c1, %32 : index
      %34 = arith.subi %33, %c1 : index
      d2m.write_col_mask_tile(%34) to %4 at %c0 of %c2 : index, !d2m.cb<memref<1x2x!ttcore.tile<32x32, f32>, #l1>>, index, index
      %35 = affine.apply #map1()[%core1]
      %36 = arith.addi %35, %c1 : index
      %37 = arith.maxsi %36, %c0 : index
      %38 = arith.shli %c1, %37 : index
      %39 = arith.subi %38, %c1 : index
      d2m.write_col_mask_tile(%39) to %4 at %c1 of %c2 : index, !d2m.cb<memref<1x2x!ttcore.tile<32x32, f32>, #l1>>, index, index
      d2m.push %4 {pin_cb = 4 : i64} : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %40 = d2m.reserve %17 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
      d2m.push %17 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %41 = arith.addi %core0, %c-1 : index
        %42 = arith.maxsi %41, %c0 : index
        %43 = arith.addi %core0, %c1 : index
        %44 = arith.minsi %43, %c9 : index
        %45 = arith.addi %core1, %c1 : index
        %46 = arith.minsi %45, %c9 : index
        %47 = arith.addi %core1, %c-1 : index
        %48 = arith.maxsi %47, %c0 : index
        %49 = d2m.wait %17 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        %50 = d2m.get_arg(18) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %50, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %50, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc[%42, %core1] into %5 srcSlice[%c9, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_load %alloc[%44, %core1] into %6 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_load %alloc[%core0, %46] into %7 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_load %alloc[%core0, %48] into %8 srcSlice[%c0, %c9] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
        %51 = d2m.wait %5 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        %52 = d2m.wait %6 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %56 = arith.cmpi slt, %arg2, %c1 : index
            %57 = affine.apply #map2(%arg3)
            %58 = affine.apply #map2(%arg2)
            %59 = d2m.reserve %9 {pin_cb = 9 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %60 = arith.muli %58, %c10 : index
            %61 = arith.addi %60, %57 : index
            %62 = arith.muli %61, %c4096 : index
            %tx_23 = d2m.dma_read %49[%62], %59[%c0], <1> {srcByteOffset = -64 : si32} : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_23 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%49, %c15) to %59 at %c16 from %61, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            scf.if %56 {
              d2m.copy_single_row_tile(%51, %c31) to %59 at %c0 from %57, %c0 : memref<1x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            } else {
              %71 = affine.apply #map3(%arg2)[%core0]
              %72 = arith.muli %71, %c10 : index
              %73 = arith.addi %72, %57 : index
              d2m.copy_single_row_tile(%49, %c31) to %59 at %c0 from %73, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            d2m.push %9 {pin_cb = 9 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %63 = arith.cmpi sge, %arg2, %c9 : index
            %64 = d2m.reserve %10 {pin_cb = 10 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %tx_24 = d2m.dma_read %49[%62], %64[%c0], <1> {srcByteOffset = 64 : si32} : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_24 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%49, %c16) to %64 at %c15 from %61, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            scf.if %63 {
              d2m.copy_single_row_tile(%52, %c0) to %64 at %c31 from %57, %c0 : memref<1x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            } else {
              %71 = affine.apply #map4(%arg2)[%core0]
              %72 = arith.muli %71, %c10 : index
              %73 = arith.addi %72, %57 : index
              d2m.copy_single_row_tile(%49, %c0) to %64 at %c31 from %73, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            d2m.push %10 {pin_cb = 10 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %65 = d2m.wait %11 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %66 = d2m.wait %12 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %67 = d2m.reserve %13 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %tx_25 = d2m.dma_read %65[%c0], %67[%c0], <1> {srcByteOffset = 64 : si32} : (memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_25 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%65, %c16) to %67 at %c15 from %c0, %c0 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            d2m.copy_single_row_tile(%66, %c0) to %67 at %c31 from %c0, %c0 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            d2m.push %13 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            d2m.pop %12 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            d2m.pop %11 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %68 = d2m.wait %14 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %69 = d2m.wait %15 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %70 = d2m.reserve %16 {pin_cb = 16 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %tx_26 = d2m.dma_read %68[%c0], %70[%c0], <1> {srcByteOffset = -64 : si32} : (memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_26 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%68, %c15) to %70 at %c16 from %c0, %c0 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            d2m.copy_single_row_tile(%69, %c31) to %70 at %c0 from %c0, %c0 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            d2m.push %16 {pin_cb = 16 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            d2m.pop %15 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            d2m.pop %14 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          }
        }
        d2m.pop %6 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %5 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        %53 = d2m.get_arg(19) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %53, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %53, %c100 reset %c0 : !d2m.local_semaphore
        %54 = d2m.reserve %17 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        %55 = d2m.wait %2 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        %tx = d2m.dma_read %55[%c0], %54[%c0], <100> : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<10x10x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx : !d2m.mem_tx<read>
        d2m.pop %2 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.push %17 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
      d2m.atomic_barrier
    }, {
    ^datamovement1:
    }, {
      %cst = arith.constant 2.000000e-01 : f32
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c9 = arith.constant 9 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c1000 = arith.constant 1000 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<2x1x!ttcore.tile<32x32, f32>, #l1>>
      %4 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(7) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
      %6 = d2m.get_cb(8) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
      %7 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %8 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %9 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %10 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %11 = d2m.get_cb(13) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %12 = d2m.get_cb(14) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %13 = d2m.get_cb(15) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %14 = d2m.get_cb(16) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %15 = d2m.get_cb(17) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %16 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %17 = d2m.reserve %15 {pin_cb = 17 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
        %18 = d2m.tile_fill(%cst) : f32 -> <32x32, f32>
        memref.store %18, %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
        %19 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
        memref.store %19, %17[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        d2m.push %15 {pin_cb = 17 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        %20 = d2m.wait %3 : <memref<2x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x1x!ttcore.tile<32x32, f32>, #l1>
        %21 = d2m.wait %4 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x2x!ttcore.tile<32x32, f32>, #l1>
        %22 = d2m.wait %16 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        %23 = d2m.wait %5 : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x1x!ttcore.tile<32x32, f32>, #l1>
        %24 = d2m.wait %6 : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x1x!ttcore.tile<32x32, f32>, #l1>
        %25 = d2m.wait %15 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %26 = d2m.reserve %2 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %27 = affine.apply #map2(%arg2)
            %28 = affine.apply #map2(%arg3)
            %29 = d2m.reserve %9 {pin_cb = 11 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %dst_23 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %30 = memref.load %22[%27, %28] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
            memref.store %30, %dst_23[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %31 = memref.load %dst_23[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %32 = "d2m.tile_transpose"(%31) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %32, %dst_23[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %33 = memref.load %dst_23[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            memref.store %33, %29[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.push %9 {pin_cb = 11 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %34 = d2m.reserve %10 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %dst_24 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %35 = memref.load %23[%27, %c0] : memref<10x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %35, %dst_24[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %36 = arith.cmpi sge, %arg3, %c9 : index
            scf.if %36 {
            } else {
              %95 = affine.apply #map4(%arg3)[%core1]
              %96 = memref.load %22[%27, %95] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
              memref.store %96, %dst_24[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            }
            %37 = memref.load %dst_24[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %38 = "d2m.tile_transpose"(%37) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %38, %dst_24[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %39 = memref.load %dst_24[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            memref.store %39, %34[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.push %10 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %40 = d2m.reserve %12 {pin_cb = 14 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %dst_25 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %41 = memref.load %22[%27, %28] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
            memref.store %41, %dst_25[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %42 = memref.load %dst_25[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %43 = "d2m.tile_transpose"(%42) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %43, %dst_25[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %44 = memref.load %dst_25[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            memref.store %44, %40[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.push %12 {pin_cb = 14 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %45 = d2m.reserve %13 {pin_cb = 15 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %dst_26 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %46 = memref.load %24[%27, %c0] : memref<10x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %46, %dst_26[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %47 = arith.cmpi slt, %arg3, %c1 : index
            scf.if %47 {
            } else {
              %95 = affine.apply #map3(%arg3)[%core1]
              %96 = memref.load %22[%27, %95] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
              memref.store %96, %dst_26[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            }
            %48 = memref.load %dst_26[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %49 = "d2m.tile_transpose"(%48) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %49, %dst_26[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %50 = memref.load %dst_26[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            memref.store %50, %45[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.push %13 {pin_cb = 15 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %dst_27 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %51 = memref.load %22[%27, %28] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
            memref.store %51, %dst_27[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %52 = memref.load %dst_27[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %53 = d2m.wait %11 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %54 = memref.load %53[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %54, %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %55 = memref.load %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %56 = d2m.wait %14 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %57 = memref.load %56[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %57, %dst_27[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %58 = memref.load %dst_27[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %59 = "d2m.tile_add"(%58, %55) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %59, %dst_27[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %60 = memref.load %dst_27[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            d2m.pop %14 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            d2m.pop %11 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %61 = "d2m.tile_transpose"(%60) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %61, %dst_27[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %62 = memref.load %dst_27[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %63 = d2m.wait %8 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %64 = memref.load %63[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %64, %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %65 = memref.load %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %66 = "d2m.tile_add"(%52, %65) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %66, %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %67 = memref.load %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            d2m.pop %8 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %68 = d2m.wait %7 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %69 = memref.load %68[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %69, %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %70 = memref.load %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %71 = "d2m.tile_add"(%67, %70) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %71, %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %72 = memref.load %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            d2m.pop %7 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %73 = "d2m.tile_add"(%72, %62) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %73, %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %74 = memref.load %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %75 = memref.load %25[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %75, %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %76 = memref.load %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %77 = "d2m.tile_mul"(%74, %76) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %77, %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %78 = memref.load %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %79 = arith.cmpi eq, %arg2, %c9 : index
            %80 = arith.select %79, %c1, %c0 : index
            %81 = affine.apply #map5(%arg2)[%core0]
            %82 = arith.cmpi slt, %81, %c0 : index
            %83 = affine.apply #map6(%arg2)[%core0]
            %84 = arith.cmpi slt, %83, %c0 : index
            %85 = arith.ori %82, %84 : i1
            %86 = scf.if %85 -> (!ttcore.tile<32x32, f32>) {
              %95 = arith.cmpi eq, %80, %c1 : index
              %96 = memref.load %20[%80, %c0] : memref<2x1x!ttcore.tile<32x32, f32>, #l1>
              memref.store %96, %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
              %97 = memref.load %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
              %98 = scf.if %95 -> (!ttcore.tile<32x32, f32>) {
                %99 = "d2m.tile_where"(%97, %78, %52) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                memref.store %99, %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                %100 = memref.load %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                scf.yield %100 : !ttcore.tile<32x32, f32>
              } else {
                %99 = "d2m.tile_where"(%97, %52, %78) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                memref.store %99, %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                %100 = memref.load %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                scf.yield %100 : !ttcore.tile<32x32, f32>
              }
              scf.yield %98 : !ttcore.tile<32x32, f32>
            } else {
              scf.yield %78 : !ttcore.tile<32x32, f32>
            }
            %87 = arith.cmpi eq, %arg3, %c9 : index
            %88 = arith.select %87, %c1, %c0 : index
            %89 = affine.apply #map5(%arg3)[%core1]
            %90 = arith.cmpi slt, %89, %c0 : index
            %91 = affine.apply #map6(%arg3)[%core1]
            %92 = arith.cmpi slt, %91, %c0 : index
            %93 = arith.ori %90, %92 : i1
            %94 = scf.if %93 -> (!ttcore.tile<32x32, f32>) {
              %95 = arith.cmpi eq, %88, %c1 : index
              %96 = memref.load %21[%c0, %88] : memref<1x2x!ttcore.tile<32x32, f32>, #l1>
              memref.store %96, %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
              %97 = memref.load %dst_27[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
              %98 = scf.if %95 -> (!ttcore.tile<32x32, f32>) {
                %99 = "d2m.tile_where"(%97, %86, %52) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                memref.store %99, %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                %100 = memref.load %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                scf.yield %100 : !ttcore.tile<32x32, f32>
              } else {
                %99 = "d2m.tile_where"(%97, %52, %86) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                memref.store %99, %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                %100 = memref.load %dst_27[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                scf.yield %100 : !ttcore.tile<32x32, f32>
              }
              scf.yield %98 : !ttcore.tile<32x32, f32>
            } else {
              scf.yield %86 : !ttcore.tile<32x32, f32>
            }
            memref.store %94, %26[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
          }
        }
        d2m.push %2 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %15 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %6 : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %5 : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %16 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
    }
    %alloc_20 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xf32, #ttcore.cb_layout<1280x4, 1>, #l1>
    %alloc_21 = memref.alloc() {address = 152064 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>)
        outs(%alloc_17 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      d2m.remote_store %alloc_17[%core0, %core1] from %2 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram> from !d2m.cb<memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
    }, {
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.reserve %2 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
      d2m.push %2 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_17 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>)
        outs(%alloc_16 : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram>)
        additionalArgs(%alloc_20, %alloc_21 : memref<32x320xf32, #ttcore.cb_layout<1280x4, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xf32, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %4 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_17[%core0, %core1] into %3 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_store %alloc_16[%core0, %core1] from %2 srcSlice[%4, %c0] : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram> from !d2m.cb<memref<32x320xf32, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %2 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xf32, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %4 = d2m.wait %2 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        %5 = d2m.reserve %3 : <memref<32x320xf32, #l1>> -> memref<32x320xf32, #l1>
        %6 = "d2m.tile_untilize_block"(%4, %5) : (memref<1x10x!ttcore.tile<32x32, f32>, #l1>, memref<32x320xf32, #l1>) -> memref<32x320xf32, #l1>
        d2m.pop %2 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.push %3 : <memref<32x320xf32, #l1>>
      }
    }
    %alloc_22 = memref.alloc() : memref<3200x3200xf32>
    d2m.to_host %alloc_16, %alloc_22 layout = #layout : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram> into memref<3200x3200xf32>
    return %alloc_22 : memref<3200x3200xf32>
  }
}
