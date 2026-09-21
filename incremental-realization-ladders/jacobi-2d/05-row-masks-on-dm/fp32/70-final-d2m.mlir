#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 3200x3200, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<()[s0] -> (s0 * 320 - 1)>
#map1 = affine_map<()[s0] -> (s0 * -320 + 2910)>
#map2 = affine_map<(d0) -> (d0 mod 10)>
#map3 = affine_map<(d0)[s0] -> (d0 + ((d0 + s0 * 10 - 1) floordiv 10) * -10 + s0 * 10 - 1)>
#map4 = affine_map<(d0)[s0] -> (d0 + ((d0 + s0 * 10 + 1) floordiv 10) * -10 + s0 * 10 + 1)>
#map5 = affine_map<()[s0] -> (s0 * -320 + 3167)>
#map6 = affine_map<()[s0] -> (s0 * 320 + 287)>
#map7 = affine_map<()[s0] -> (s0 * -320 + 2879)>
#map8 = affine_map<(d0)[s0] -> (d0 * 32 + s0 * 320 - 1)>
#map9 = affine_map<(d0)[s0] -> (d0 * -32 - s0 * 320 + 3167)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xf32>) -> memref<3200x3200xf32> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 1445888 : i64, tt.l1_persistent_base = 626688 : i64, tt.l1_pool = 626688 : i64} {
    %alloc = memref.alloc() {address = 737792 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 1147392 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<10x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 520704 : i64, alignment = 16 : i64} : memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<8192x4096, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 528896 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 569856 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 610816 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_6 = memref.alloc() {address = 651776 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_7 = memref.alloc() {address = 692736 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_8 = memref.alloc() {address = 700928 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_9 = memref.alloc() {address = 709120 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_10 = memref.alloc() {address = 717312 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_11 = memref.alloc() {address = 725504 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_12 = memref.alloc() {address = 733696 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_13 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram>
    %alloc_14 = memref.alloc() {address = 49348608 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>
    d2m.to_device %arg0, %alloc_13 layout = #layout : memref<3200x3200xf32> into memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram>
    %alloc_15 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xf32, #ttcore.cb_layout<1280x4, 1>, #l1>
    %alloc_16 = memref.alloc() {address = 152064 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_13 : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram>)
        outs(%alloc_14 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>)
        additionalArgs(%alloc_15, %alloc_16 : memref<32x320xf32, #ttcore.cb_layout<1280x4, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xf32, #l1>>
      %5 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %6 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_13[%core0, %core1] into %4 srcSlice[%6, %c0] : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram> into !d2m.cb<memref<32x320xf32, #l1>>
        d2m.remote_store %alloc_14[%core0, %core1] from %5 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram> from !d2m.cb<memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %4 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xf32, #l1>>
      %5 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %6 = d2m.wait %4 : <memref<32x320xf32, #l1>> -> memref<32x320xf32, #l1>
        %7 = d2m.reserve %5 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        %8 = "d2m.tile_tilize_block"(%6, %7) : (memref<32x320xf32, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #l1>) -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        d2m.pop %4 : <memref<32x320xf32, #l1>>
        d2m.push %5 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      }
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_14 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      d2m.remote_load %alloc_14[%core0, %core1] into %4 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0:
    }
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %2 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %3 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc, %alloc_0 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>, memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>)
        additionalArgs(%alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %alloc_6, %alloc_7, %alloc_8, %alloc_9, %alloc_10, %alloc_11, %alloc_12, %0, %1, %2, %3 : memref<10x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>, memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<8192x4096, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c90 = arith.constant 90 : index
      %c40960 = arith.constant 40960 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c9 = arith.constant 9 : index
      %c-1 = arith.constant -1 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c4096 = arith.constant 4096 : index
      %c15 = arith.constant 15 : index
      %c16 = arith.constant 16 : index
      %c31 = arith.constant 31 : index
      %c1000 = arith.constant 1000 : index
      %4 = d2m.get_cb(3) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %6 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      %7 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      %8 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %9 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %10 = d2m.get_cb(13) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %11 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %12 = d2m.reserve %5 {pin_cb = 4 : i64} : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x2x!ttcore.tile<32x32, f32>, #l1>
      %13 = affine.apply #map()[%core1]
      %14 = arith.subi %c0, %13 : index
      %15 = arith.maxsi %14, %c0 : index
      %16 = arith.shli %c1, %15 : index
      %17 = arith.subi %16, %c1 : index
      d2m.write_col_mask_tile(%17) to %5 at %c0 of %c2 : index, !d2m.cb<memref<1x2x!ttcore.tile<32x32, f32>, #l1>>, index, index
      %18 = affine.apply #map1()[%core1]
      %19 = arith.addi %18, %c1 : index
      %20 = arith.maxsi %19, %c0 : index
      %21 = arith.shli %c1, %20 : index
      %22 = arith.subi %21, %c1 : index
      d2m.write_col_mask_tile(%22) to %5 at %c1 of %c2 : index, !d2m.cb<memref<1x2x!ttcore.tile<32x32, f32>, #l1>>, index, index
      d2m.push %5 {pin_cb = 4 : i64} : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %23 = d2m.reserve %11 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
      d2m.push %11 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %24 = arith.addi %core0, %c-1 : index
        %25 = arith.maxsi %24, %c0 : index
        %26 = arith.addi %core0, %c1 : index
        %27 = arith.minsi %26, %c9 : index
        %28 = d2m.wait %11 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        %29 = d2m.get_arg(15) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %29, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %29, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc[%25, %core1] into %6 srcSlice[%c9, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_load %alloc[%27, %core1] into %7 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        %30 = d2m.wait %6 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        %31 = d2m.wait %7 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %35 = arith.cmpi slt, %arg2, %c1 : index
            %36 = affine.apply #map2(%arg3)
            %37 = affine.apply #map2(%arg2)
            %38 = d2m.reserve %8 {pin_cb = 9 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %39 = arith.muli %37, %c10 : index
            %40 = arith.addi %39, %36 : index
            %41 = arith.muli %40, %c4096 : index
            %tx = d2m.dma_read %28[%41], %38[%c0], <1> {srcByteOffset = -64 : si32} : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%28, %c15) to %38 at %c16 from %40, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            scf.if %35 {
              d2m.copy_single_row_tile(%30, %c31) to %38 at %c0 from %36, %c0 : memref<1x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            } else {
              %44 = affine.apply #map3(%arg2)[%core0]
              %45 = arith.muli %44, %c10 : index
              %46 = arith.addi %45, %36 : index
              d2m.copy_single_row_tile(%28, %c31) to %38 at %c0 from %46, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            d2m.push %8 {pin_cb = 9 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %42 = arith.cmpi sge, %arg2, %c9 : index
            %43 = d2m.reserve %9 {pin_cb = 10 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %tx_20 = d2m.dma_read %28[%41], %43[%c0], <1> {srcByteOffset = 64 : si32} : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_20 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%28, %c16) to %43 at %c15 from %40, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            scf.if %42 {
              d2m.copy_single_row_tile(%31, %c0) to %43 at %c31 from %36, %c0 : memref<1x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            } else {
              %44 = affine.apply #map4(%arg2)[%core0]
              %45 = arith.muli %44, %c10 : index
              %46 = arith.addi %45, %36 : index
              d2m.copy_single_row_tile(%28, %c0) to %43 at %c31 from %46, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            d2m.push %9 {pin_cb = 10 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          }
        }
        d2m.pop %7 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %6 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        %32 = d2m.get_arg(17) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %32, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %32, %c100 reset %c0 : !d2m.local_semaphore
        %33 = d2m.reserve %11 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        %34 = d2m.wait %4 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %35 = affine.apply #map2(%arg2)
          %36 = affine.apply #map()[%core0]
          %37 = arith.cmpi slt, %36, %c0 : index
          %38 = affine.apply #map5()[%core0]
          %39 = arith.cmpi slt, %38, %c0 : index
          %40 = arith.ori %37, %39 : i1
          scf.if %40 {
            %41 = d2m.reserve %10 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %42 = arith.muli %35, %c4096 : index
            %tx = d2m.dma_read %34[%42], %41[%c0], <1> : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
            scf.if %37 {
              d2m.copy_single_row_tile(%33, %c0) to %41 at %c0 from %35, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            scf.if %39 {
              d2m.copy_single_row_tile(%33, %c31) to %41 at %c31 from %35, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            d2m.push %10 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %43 = d2m.wait %10 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %44 = arith.muli %arg2, %c4096 : index
            %tx_20 = d2m.dma_read %43[%c0], %33[%44], <1> : (memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<10x10x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_20 : !d2m.mem_tx<read>
            d2m.pop %10 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          } else {
            %41 = arith.muli %arg2, %c4096 : index
            %42 = arith.muli %35, %c4096 : index
            %tx = d2m.dma_read %34[%42], %33[%41], <1> : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<10x10x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
          }
        }
        scf.for %arg2 = %c1 to %c9 step %c1 {
          %35 = affine.apply #map2(%arg2)
          %36 = arith.muli %arg2, %c40960 : index
          %37 = arith.muli %35, %c40960 : index
          %tx = d2m.dma_read %34[%37], %33[%36], <10> : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<10x10x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx : !d2m.mem_tx<read>
        }
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %35 = affine.apply #map2(%arg2)
          %36 = affine.apply #map6()[%core0]
          %37 = arith.cmpi slt, %36, %c0 : index
          %38 = affine.apply #map7()[%core0]
          %39 = arith.cmpi slt, %38, %c0 : index
          %40 = arith.ori %37, %39 : i1
          scf.if %40 {
            %41 = d2m.reserve %10 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %42 = arith.addi %35, %c90 : index
            %43 = arith.muli %42, %c4096 : index
            %tx = d2m.dma_read %34[%43], %41[%c0], <1> : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
            scf.if %37 {
              d2m.copy_single_row_tile(%33, %c0) to %41 at %c0 from %42, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            scf.if %39 {
              d2m.copy_single_row_tile(%33, %c31) to %41 at %c31 from %42, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            d2m.push %10 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %44 = d2m.wait %10 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %45 = arith.addi %arg2, %c90 : index
            %46 = arith.muli %45, %c4096 : index
            %tx_20 = d2m.dma_read %44[%c0], %33[%46], <1> : (memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<10x10x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_20 : !d2m.mem_tx<read>
            d2m.pop %10 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          } else {
            %41 = arith.addi %arg2, %c90 : index
            %42 = arith.addi %35, %c90 : index
            %43 = arith.muli %41, %c4096 : index
            %44 = arith.muli %42, %c4096 : index
            %tx = d2m.dma_read %34[%44], %33[%43], <1> : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<10x10x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
          }
        }
        d2m.pop %4 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.push %11 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
      d2m.atomic_barrier
    }, {
      %c1 = arith.constant 1 : index
      %c9 = arith.constant 9 : index
      %c-1 = arith.constant -1 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c4096 = arith.constant 4096 : index
      %c16 = arith.constant 16 : index
      %c15 = arith.constant 15 : index
      %c31 = arith.constant 31 : index
      %c1000 = arith.constant 1000 : index
      %4 = d2m.get_cb(7) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(8) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
      %6 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %7 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %8 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %9 = arith.addi %core1, %c1 : index
        %10 = arith.minsi %9, %c9 : index
        %11 = arith.addi %core1, %c-1 : index
        %12 = arith.maxsi %11, %c0 : index
        %13 = d2m.wait %8 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        %14 = d2m.get_arg(16) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %14, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %14, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc_0[%core0, %10] into %4 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_load %alloc_0[%core0, %12] into %5 srcSlice[%c0, %c9] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
        %15 = d2m.wait %4 : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x1x!ttcore.tile<32x32, f32>, #l1>
        %16 = d2m.wait %5 : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x1x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %18 = affine.apply #map2(%arg2)
            %19 = affine.apply #map2(%arg3)
            %20 = arith.cmpi sge, %arg3, %c9 : index
            %21 = d2m.reserve %6 {pin_cb = 11 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %22 = arith.muli %18, %c10 : index
            %23 = arith.addi %22, %19 : index
            %24 = arith.muli %23, %c4096 : index
            %tx = d2m.dma_read %13[%24], %21[%c0], <1> {srcByteOffset = 64 : si32} : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%13, %c16) to %21 at %c15 from %23, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            scf.if %20 {
              d2m.copy_single_row_tile(%15, %c0) to %21 at %c31 from %18, %c0 : memref<10x1x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            } else {
              %27 = affine.apply #map4(%arg3)[%core1]
              %28 = arith.addi %22, %27 : index
              d2m.copy_single_row_tile(%13, %c0) to %21 at %c31 from %28, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            d2m.push %6 {pin_cb = 11 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %25 = arith.cmpi slt, %arg3, %c1 : index
            %26 = d2m.reserve %7 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %tx_20 = d2m.dma_read %13[%24], %26[%c0], <1> {srcByteOffset = -64 : si32} : (memref<10x10x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_20 : !d2m.mem_tx<read>
            d2m.copy_single_row_tile(%13, %c15) to %26 at %c16 from %23, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            scf.if %25 {
              d2m.copy_single_row_tile(%16, %c31) to %26 at %c0 from %18, %c0 : memref<10x1x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            } else {
              %27 = affine.apply #map3(%arg3)[%core1]
              %28 = arith.addi %22, %27 : index
              d2m.copy_single_row_tile(%13, %c31) to %26 at %c0 from %28, %c0 : memref<10x10x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
            }
            d2m.push %7 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          }
        }
        d2m.pop %5 : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %4 : <memref<10x1x!ttcore.tile<32x32, f32>, #l1>>
        %17 = d2m.get_arg(18) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %17, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %17, %c100 reset %c0 : !d2m.local_semaphore
        d2m.pop %8 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
      d2m.atomic_barrier
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 2.000000e-01 : f32
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c9 = arith.constant 9 : index
      %c1000 = arith.constant 1000 : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(3) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %6 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %7 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %8 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %9 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %10 = d2m.get_cb(14) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %11 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %12 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %13 = d2m.wait %12 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        %14 = d2m.reserve %11 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %dst_20 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %21 = memref.load %13[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
            memref.store %21, %dst_20[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %22 = memref.load %dst_20[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %23 = "d2m.tile_transpose"(%22) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %23, %dst_20[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %24 = memref.load %dst_20[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            memref.store %24, %14[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
          }
        }
        d2m.push %11 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
        %15 = d2m.reserve %10 {pin_cb = 14 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
        %16 = d2m.tile_fill(%cst) : f32 -> <32x32, f32>
        memref.store %16, %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
        %17 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
        memref.store %17, %15[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        d2m.push %10 {pin_cb = 14 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        %18 = d2m.wait %5 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x2x!ttcore.tile<32x32, f32>, #l1>
        %19 = d2m.wait %10 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %20 = d2m.reserve %4 {pin_cb = 3 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %21 = affine.apply #map2(%arg2)
            %22 = affine.apply #map2(%arg3)
            %dst_20 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %23 = memref.load %13[%21, %22] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
            memref.store %23, %dst_20[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %24 = memref.load %dst_20[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %25 = d2m.wait %8 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %26 = memref.load %25[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %26, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %27 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %28 = d2m.wait %9 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %29 = memref.load %28[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %29, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %30 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %31 = "d2m.tile_add"(%30, %27) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %31, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %32 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            d2m.pop %9 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            d2m.pop %8 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %33 = "d2m.tile_transpose"(%32) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %33, %dst_20[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %34 = memref.load %dst_20[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %35 = d2m.wait %7 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %36 = memref.load %35[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %36, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %37 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %38 = "d2m.tile_add"(%24, %37) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %38, %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %39 = memref.load %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            d2m.pop %7 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %40 = d2m.wait %6 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            %41 = memref.load %40[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %41, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %42 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %43 = "d2m.tile_add"(%39, %42) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %43, %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %44 = memref.load %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            d2m.pop %6 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
            %45 = "d2m.tile_add"(%44, %34) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %45, %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %46 = memref.load %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %47 = memref.load %19[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %47, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %48 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %49 = "d2m.tile_mul"(%46, %48) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %49, %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %50 = memref.load %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %51 = arith.cmpi eq, %arg3, %c9 : index
            %52 = arith.select %51, %c1, %c0 : index
            %53 = affine.apply #map8(%arg3)[%core1]
            %54 = arith.cmpi slt, %53, %c0 : index
            %55 = affine.apply #map9(%arg3)[%core1]
            %56 = arith.cmpi slt, %55, %c0 : index
            %57 = arith.ori %54, %56 : i1
            %58 = scf.if %57 -> (!ttcore.tile<32x32, f32>) {
              %59 = arith.cmpi eq, %52, %c1 : index
              %60 = memref.load %18[%c0, %52] : memref<1x2x!ttcore.tile<32x32, f32>, #l1>
              memref.store %60, %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
              %61 = memref.load %dst_20[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
              %62 = scf.if %59 -> (!ttcore.tile<32x32, f32>) {
                %63 = "d2m.tile_where"(%61, %50, %24) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                memref.store %63, %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                %64 = memref.load %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                scf.yield %64 : !ttcore.tile<32x32, f32>
              } else {
                %63 = "d2m.tile_where"(%61, %24, %50) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                memref.store %63, %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                %64 = memref.load %dst_20[%c3] : memref<16x!ttcore.tile<32x32, f32>, #dst>
                scf.yield %64 : !ttcore.tile<32x32, f32>
              }
              scf.yield %62 : !ttcore.tile<32x32, f32>
            } else {
              scf.yield %50 : !ttcore.tile<32x32, f32>
            }
            memref.store %58, %20[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, f32>, #l1>
          }
        }
        d2m.push %4 {pin_cb = 3 : i64} : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %10 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %12 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
    }
    %alloc_17 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xf32, #ttcore.cb_layout<1280x4, 1>, #l1>
    %alloc_18 = memref.alloc() {address = 152064 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #l1>)
        outs(%alloc_14 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      d2m.remote_store %alloc_14[%core0, %core1] from %4 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram> from !d2m.cb<memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
    }, {
      %4 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.reserve %4 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<10x10x!ttcore.tile<32x32, f32>, #l1>
      d2m.push %4 : <memref<10x10x!ttcore.tile<32x32, f32>, #l1>>
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_14 : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram>)
        outs(%alloc_13 : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram>)
        additionalArgs(%alloc_17, %alloc_18 : memref<32x320xf32, #ttcore.cb_layout<1280x4, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<40960x4096, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xf32, #l1>>
      %5 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %6 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_14[%core0, %core1] into %5 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, f32>, #ttcore.shard<40960x4096, 1>, #dram> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_store %alloc_13[%core0, %core1] from %4 srcSlice[%6, %c0] : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram> from !d2m.cb<memref<32x320xf32, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %4 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xf32, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %6 = d2m.wait %4 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x10x!ttcore.tile<32x32, f32>, #l1>
        %7 = d2m.reserve %5 : <memref<32x320xf32, #l1>> -> memref<32x320xf32, #l1>
        %8 = "d2m.tile_untilize_block"(%6, %7) : (memref<1x10x!ttcore.tile<32x32, f32>, #l1>, memref<32x320xf32, #l1>) -> memref<32x320xf32, #l1>
        d2m.pop %4 : <memref<1x10x!ttcore.tile<32x32, f32>, #l1>>
        d2m.push %5 : <memref<32x320xf32, #l1>>
      }
    }
    %alloc_19 = memref.alloc() : memref<3200x3200xf32>
    d2m.to_host %alloc_13, %alloc_19 layout = #layout : memref<10x10x320x320xf32, #ttcore.shard<1280x4, 1>, #dram> into memref<3200x3200xf32>
    return %alloc_19 : memref<3200x3200xf32>
  }
}
