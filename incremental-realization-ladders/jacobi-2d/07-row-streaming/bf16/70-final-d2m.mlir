#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 3200x3200, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0) -> (d0 mod 10)>
#map1 = affine_map<(d0)[s0] -> (d0 + ((d0 + s0 * 10 - 1) floordiv 10) * -10 + s0 * 10 - 1)>
#map2 = affine_map<(d0)[s0] -> (d0 + ((d0 + s0 * 10 + 1) floordiv 10) * -10 + s0 * 10 + 1)>
#map3 = affine_map<()[s0] -> (s0 * 320 - 1)>
#map4 = affine_map<()[s0] -> (s0 * -320 + 3167)>
#map5 = affine_map<(d0)[s0] -> (d0 * 32 + s0 * 320 - 1)>
#map6 = affine_map<(d0)[s0] -> (d0 * -32 - s0 * 320 + 3167)>
#map7 = affine_map<()[s0] -> (s0 * 320 + 287)>
#map8 = affine_map<()[s0] -> (s0 * -320 + 2879)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xbf16>) -> memref<3200x3200xbf16> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 866304 : i64, tt.l1_persistent_base = 456704 : i64, tt.l1_pool = 456704 : i64} {
    %alloc = memref.alloc() {address = 567808 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 772608 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 315904 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 336384 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 356864 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 377344 : i64, alignment = 16 : i64} : memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_6 = memref.alloc() {address = 397824 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_7 = memref.alloc() {address = 438784 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_8 = memref.alloc() {address = 479744 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_9 = memref.alloc() {address = 520704 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_10 = memref.alloc() {address = 561664 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_11 = memref.alloc() {address = 565760 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_12 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_13 = memref.alloc() {address = 28868608 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>
    d2m.to_device %arg0, %alloc_12 layout = #layout : memref<3200x3200xbf16> into memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_14 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_15 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_12 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        outs(%alloc_13 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        additionalArgs(%alloc_14, %alloc_15 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %5 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %6 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_12[%core0, %core1] into %4 srcSlice[%6, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into !d2m.cb<memref<32x320xbf16, #l1>>
        d2m.remote_store %alloc_13[%core0, %core1] from %5 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %4 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %5 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %6 = d2m.wait %4 : <memref<32x320xbf16, #l1>> -> memref<32x320xbf16, #l1>
        %7 = d2m.reserve %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %8 = "d2m.tile_tilize_block"(%6, %7) : (memref<32x320xbf16, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        d2m.pop %4 : <memref<32x320xbf16, #l1>>
        d2m.push %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_13 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_load %alloc_13[%core0, %core1] into %4 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
    ^compute0:
    }
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %2 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %3 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc, %alloc_0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>, memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        additionalArgs(%alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %alloc_6, %alloc_7, %alloc_8, %alloc_9, %alloc_10, %alloc_11, %0, %1, %2, %3 : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<10x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c90 = arith.constant 90 : index
      %c20480 = arith.constant 20480 : index
      %c-1 = arith.constant -1 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9 = arith.constant 9 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c2048 = arith.constant 2048 : index
      %c15 = arith.constant 15 : index
      %c16 = arith.constant 16 : index
      %c31 = arith.constant 31 : index
      %c1000 = arith.constant 1000 : index
      %4 = d2m.get_cb(3) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %7 = d2m.get_cb(8) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %8 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %9 = d2m.get_cb(12) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %10 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %11 = d2m.reserve %10 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %10 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %12 = arith.addi %core0, %c-1 : index
        %13 = arith.maxsi %12, %c0 : index
        %14 = arith.addi %core0, %c1 : index
        %15 = arith.minsi %14, %c9 : index
        %16 = d2m.wait %10 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %17 = d2m.get_arg(14) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %17, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %17, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc[%13, %core1] into %5 srcSlice[%c9, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc[%15, %core1] into %6 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        %18 = d2m.wait %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %19 = d2m.wait %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %23 = d2m.reserve %7 {pin_cb = 8 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %24 = d2m.reserve %8 {pin_cb = 9 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %25 = affine.apply #map(%arg2)
          %26 = arith.muli %25, %c20480 : index
          %tx = d2m.dma_read %16[%26], %23[%c0], <10> {srcByteOffset = -32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx : !d2m.mem_tx<read>
          %tx_19 = d2m.dma_read %16[%26], %24[%c0], <10> {srcByteOffset = 32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx_19 : !d2m.mem_tx<read>
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %27 = arith.cmpi slt, %arg2, %c1 : index
            %28 = affine.apply #map(%arg3)
            %29 = arith.muli %25, %c10 : index
            %30 = arith.addi %29, %28 : index
            d2m.copy_single_row_tile(%16, %c15) to %23 at %c16 from %30, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %27 {
              d2m.copy_single_row_tile(%18, %c31) to %23 at %c0 from %28, %arg3 : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %32 = affine.apply #map1(%arg2)[%core0]
              %33 = arith.muli %32, %c10 : index
              %34 = arith.addi %33, %28 : index
              d2m.copy_single_row_tile(%16, %c31) to %23 at %c0 from %34, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            %31 = arith.cmpi sge, %arg2, %c9 : index
            d2m.copy_single_row_tile(%16, %c16) to %24 at %c15 from %30, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %31 {
              d2m.copy_single_row_tile(%19, %c0) to %24 at %c31 from %28, %arg3 : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %32 = affine.apply #map2(%arg2)[%core0]
              %33 = arith.muli %32, %c10 : index
              %34 = arith.addi %33, %28 : index
              d2m.copy_single_row_tile(%16, %c0) to %24 at %c31 from %34, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
          }
          d2m.push %8 {pin_cb = 9 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.push %7 {pin_cb = 8 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        }
        d2m.pop %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        %20 = d2m.get_arg(16) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %20, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %20, %c100 reset %c0 : !d2m.local_semaphore
        %21 = d2m.reserve %10 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %22 = d2m.wait %4 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %23 = affine.apply #map(%arg2)
          %24 = affine.apply #map3()[%core0]
          %25 = arith.cmpi slt, %24, %c0 : index
          %26 = affine.apply #map4()[%core0]
          %27 = arith.cmpi slt, %26, %c0 : index
          %28 = arith.ori %25, %27 : i1
          %29 = affine.apply #map5(%arg2)[%core1]
          %30 = arith.cmpi slt, %29, %c0 : index
          %31 = arith.ori %28, %30 : i1
          %32 = affine.apply #map6(%arg2)[%core1]
          %33 = arith.cmpi slt, %32, %c0 : index
          %34 = arith.ori %31, %33 : i1
          scf.if %34 {
            %35 = d2m.reserve %9 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %36 = arith.muli %23, %c2048 : index
            %tx = d2m.dma_read %22[%36], %35[%c0], <1> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
            scf.if %25 {
              d2m.copy_single_row_tile(%21, %c0) to %35 at %c0 from %23, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %27 {
              d2m.copy_single_row_tile(%21, %c31) to %35 at %c31 from %23, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %30 {
              d2m.copy_single_col_tile(%21, %c0) to %35 at %c0 from %23, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %33 {
              d2m.copy_single_col_tile(%21, %c31) to %35 at %c31 from %23, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            d2m.push %9 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %37 = d2m.wait %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %38 = arith.muli %arg2, %c2048 : index
            %tx_19 = d2m.dma_read %37[%c0], %21[%38], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_19 : !d2m.mem_tx<read>
            d2m.pop %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
          } else {
            %35 = arith.muli %arg2, %c2048 : index
            %36 = arith.muli %23, %c2048 : index
            %tx = d2m.dma_read %22[%36], %21[%35], <1> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
          }
        }
        scf.for %arg2 = %c1 to %c9 step %c1 {
          %23 = affine.apply #map(%arg2)
          %24 = affine.apply #map5(%arg2)[%core0]
          %25 = arith.cmpi slt, %24, %c0 : index
          %26 = affine.apply #map6(%arg2)[%core0]
          %27 = arith.cmpi slt, %26, %c0 : index
          %28 = arith.ori %25, %27 : i1
          %29 = affine.apply #map3()[%core1]
          %30 = arith.cmpi slt, %29, %c0 : index
          %31 = arith.ori %28, %30 : i1
          %32 = affine.apply #map4()[%core1]
          %33 = arith.cmpi slt, %32, %c0 : index
          %34 = arith.ori %31, %33 : i1
          scf.if %34 {
            %47 = d2m.reserve %9 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %48 = arith.muli %23, %c20480 : index
            %tx_19 = d2m.dma_read %22[%48], %47[%c0], <1> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_19 : !d2m.mem_tx<read>
            scf.if %25 {
              %51 = arith.muli %23, %c10 : index
              d2m.copy_single_row_tile(%21, %c0) to %47 at %c0 from %51, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %27 {
              %51 = arith.muli %23, %c10 : index
              d2m.copy_single_row_tile(%21, %c31) to %47 at %c31 from %51, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %30 {
              %51 = arith.muli %23, %c10 : index
              d2m.copy_single_col_tile(%21, %c0) to %47 at %c0 from %51, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %33 {
              %51 = arith.muli %23, %c10 : index
              d2m.copy_single_col_tile(%21, %c31) to %47 at %c31 from %51, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            d2m.push %9 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %49 = d2m.wait %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %50 = arith.muli %arg2, %c20480 : index
            %tx_20 = d2m.dma_read %49[%c0], %21[%50], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_20 : !d2m.mem_tx<read>
            d2m.pop %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
          } else {
            %47 = arith.muli %arg2, %c20480 : index
            %48 = arith.muli %23, %c20480 : index
            %tx_19 = d2m.dma_read %22[%48], %21[%47], <1> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_19 : !d2m.mem_tx<read>
          }
          %35 = arith.muli %arg2, %c10 : index
          %36 = arith.addi %35, %c1 : index
          %37 = arith.muli %23, %c10 : index
          %38 = arith.addi %37, %c1 : index
          %39 = arith.muli %36, %c2048 : index
          %40 = arith.muli %38, %c2048 : index
          %tx = d2m.dma_read %22[%40], %21[%39], <8> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx : !d2m.mem_tx<read>
          %41 = affine.apply #map7()[%core1]
          %42 = arith.cmpi slt, %41, %c0 : index
          %43 = arith.ori %28, %42 : i1
          %44 = affine.apply #map8()[%core1]
          %45 = arith.cmpi slt, %44, %c0 : index
          %46 = arith.ori %43, %45 : i1
          scf.if %46 {
            %47 = d2m.reserve %9 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %48 = arith.addi %37, %c9 : index
            %49 = arith.muli %48, %c2048 : index
            %tx_19 = d2m.dma_read %22[%49], %47[%c0], <1> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_19 : !d2m.mem_tx<read>
            scf.if %25 {
              d2m.copy_single_row_tile(%21, %c0) to %47 at %c0 from %48, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %27 {
              d2m.copy_single_row_tile(%21, %c31) to %47 at %c31 from %48, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %42 {
              d2m.copy_single_col_tile(%21, %c0) to %47 at %c0 from %48, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %45 {
              d2m.copy_single_col_tile(%21, %c31) to %47 at %c31 from %48, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            d2m.push %9 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %50 = d2m.wait %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %51 = arith.addi %35, %c9 : index
            %52 = arith.muli %51, %c2048 : index
            %tx_20 = d2m.dma_read %50[%c0], %21[%52], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_20 : !d2m.mem_tx<read>
            d2m.pop %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
          } else {
            %47 = arith.addi %35, %c9 : index
            %48 = arith.addi %37, %c9 : index
            %49 = arith.muli %47, %c2048 : index
            %50 = arith.muli %48, %c2048 : index
            %tx_19 = d2m.dma_read %22[%50], %21[%49], <1> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_19 : !d2m.mem_tx<read>
          }
        }
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %23 = affine.apply #map(%arg2)
          %24 = affine.apply #map7()[%core0]
          %25 = arith.cmpi slt, %24, %c0 : index
          %26 = affine.apply #map8()[%core0]
          %27 = arith.cmpi slt, %26, %c0 : index
          %28 = arith.ori %25, %27 : i1
          %29 = affine.apply #map5(%arg2)[%core1]
          %30 = arith.cmpi slt, %29, %c0 : index
          %31 = arith.ori %28, %30 : i1
          %32 = affine.apply #map6(%arg2)[%core1]
          %33 = arith.cmpi slt, %32, %c0 : index
          %34 = arith.ori %31, %33 : i1
          scf.if %34 {
            %35 = d2m.reserve %9 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %36 = arith.addi %23, %c90 : index
            %37 = arith.muli %36, %c2048 : index
            %tx = d2m.dma_read %22[%37], %35[%c0], <1> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
            scf.if %25 {
              d2m.copy_single_row_tile(%21, %c0) to %35 at %c0 from %36, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %27 {
              d2m.copy_single_row_tile(%21, %c31) to %35 at %c31 from %36, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %30 {
              d2m.copy_single_col_tile(%21, %c0) to %35 at %c0 from %36, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            scf.if %33 {
              d2m.copy_single_col_tile(%21, %c31) to %35 at %c31 from %36, %c0 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            d2m.push %9 {pin_cb = 12 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %38 = d2m.wait %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %39 = arith.addi %arg2, %c90 : index
            %40 = arith.muli %39, %c2048 : index
            %tx_19 = d2m.dma_read %38[%c0], %21[%40], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx_19 : !d2m.mem_tx<read>
            d2m.pop %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
          } else {
            %35 = arith.addi %arg2, %c90 : index
            %36 = arith.addi %23, %c90 : index
            %37 = arith.muli %35, %c2048 : index
            %38 = arith.muli %36, %c2048 : index
            %tx = d2m.dma_read %22[%38], %21[%37], <1> : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
            d2m.dma_wait %tx : !d2m.mem_tx<read>
          }
        }
        d2m.pop %4 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.push %10 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
      d2m.atomic_barrier
    }, {
      %c20480 = arith.constant 20480 : index
      %c1 = arith.constant 1 : index
      %c9 = arith.constant 9 : index
      %c-1 = arith.constant -1 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c16 = arith.constant 16 : index
      %c15 = arith.constant 15 : index
      %c31 = arith.constant 31 : index
      %c1000 = arith.constant 1000 : index
      %4 = d2m.get_cb(6) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(7) resolution_stage =  compile : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %7 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %8 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %9 = arith.addi %core1, %c1 : index
        %10 = arith.minsi %9, %c9 : index
        %11 = arith.addi %core1, %c-1 : index
        %12 = arith.maxsi %11, %c0 : index
        %13 = d2m.wait %8 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %14 = d2m.get_arg(15) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %14, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %14, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc_0[%core0, %10] into %4 srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc_0[%core0, %12] into %5 srcSlice[%c0, %c9] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        %15 = d2m.wait %4 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x1x!ttcore.tile<32x32, bf16>, #l1>
        %16 = d2m.wait %5 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x1x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %18 = d2m.reserve %6 {pin_cb = 10 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %19 = d2m.reserve %7 {pin_cb = 11 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %20 = affine.apply #map(%arg2)
          %21 = arith.muli %20, %c20480 : index
          %tx = d2m.dma_read %13[%21], %18[%c0], <10> {srcByteOffset = 32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx : !d2m.mem_tx<read>
          %tx_19 = d2m.dma_read %13[%21], %19[%c0], <10> {srcByteOffset = -32 : si32} : (memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx_19 : !d2m.mem_tx<read>
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %22 = affine.apply #map(%arg3)
            %23 = arith.cmpi sge, %arg3, %c9 : index
            %24 = arith.muli %20, %c10 : index
            %25 = arith.addi %24, %22 : index
            d2m.copy_single_row_tile(%13, %c16) to %18 at %c15 from %25, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %23 {
              d2m.copy_single_row_tile(%15, %c0) to %18 at %c31 from %20, %arg3 : memref<10x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %27 = affine.apply #map2(%arg3)[%core1]
              %28 = arith.addi %24, %27 : index
              d2m.copy_single_row_tile(%13, %c0) to %18 at %c31 from %28, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
            %26 = arith.cmpi slt, %arg3, %c1 : index
            d2m.copy_single_row_tile(%13, %c15) to %19 at %c16 from %25, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            scf.if %26 {
              d2m.copy_single_row_tile(%16, %c31) to %19 at %c0 from %20, %arg3 : memref<10x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            } else {
              %27 = affine.apply #map1(%arg3)[%core1]
              %28 = arith.addi %24, %27 : index
              d2m.copy_single_row_tile(%13, %c31) to %19 at %c0 from %28, %arg3 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
            }
          }
          d2m.push %7 {pin_cb = 11 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.push %6 {pin_cb = 10 : i64} : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        }
        d2m.pop %5 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %4 : <memref<10x1x!ttcore.tile<32x32, bf16>, #l1>>
        %17 = d2m.get_arg(17) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %17, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %17, %c100 reset %c0 : !d2m.local_semaphore
        d2m.pop %8 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
      d2m.atomic_barrier
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 2.001950e-01 : bf16
      %c1000 = arith.constant 1000 : index
      %4 = d2m.get_cb(3) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(8) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %7 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %8 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %9 = d2m.get_cb(13) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %10 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %11 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c1000 step %c1 {
        %12 = d2m.wait %11 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        %13 = d2m.reserve %10 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %dst_19 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %19 = memref.load %12[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            memref.store %19, %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %20 = memref.load %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %21 = "d2m.tile_transpose"(%20) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %21, %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %22 = memref.load %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            memref.store %22, %13[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          }
        }
        d2m.push %10 {pin_cb = 1 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        %14 = d2m.reserve %9 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        %15 = d2m.tile_fill(%cst) : bf16 -> <32x32, bf16>
        memref.store %15, %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        %16 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
        memref.store %16, %14[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.push %9 {pin_cb = 13 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        %17 = d2m.wait %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        %18 = d2m.reserve %4 {pin_cb = 3 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %19 = d2m.wait %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %20 = d2m.wait %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %21 = d2m.wait %7 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          %22 = d2m.wait %8 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %23 = affine.apply #map(%arg2)
            %24 = affine.apply #map(%arg3)
            %25 = memref.load %12[%23, %24] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            %26 = memref.load %21[%c0, %arg3] : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
            %27 = memref.load %22[%c0, %arg3] : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
            %dst_19 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %28 = "d2m.tile_add"(%27, %26) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %28, %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %29 = memref.load %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %30 = "d2m.tile_transpose"(%29) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %30, %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %31 = memref.load %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %32 = memref.load %20[%c0, %arg3] : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
            %33 = "d2m.tile_add"(%25, %32) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %33, %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %34 = memref.load %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %35 = memref.load %19[%c0, %arg3] : memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
            %36 = "d2m.tile_add"(%34, %35) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %36, %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %37 = memref.load %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %38 = "d2m.tile_add"(%37, %31) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %38, %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %39 = memref.load %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %40 = memref.load %17[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %41 = "d2m.tile_mul"(%39, %40) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %41, %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %42 = memref.load %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            memref.store %42, %18[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          }
          d2m.pop %8 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.pop %7 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.pop %6 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.pop %5 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        }
        d2m.push %4 {pin_cb = 3 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %9 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %11 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 1000 : i64}
    }
    %alloc_16 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_17 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc_13 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_store %alloc_13[%core0, %core1] from %4 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
      %4 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.reserve %4 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %4 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_13 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc_12 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        additionalArgs(%alloc_16, %alloc_17 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %4 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %5 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %6 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_13[%core0, %core1] into %5 srcSlice[%arg1, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_store %alloc_12[%core0, %core1] from %4 srcSlice[%6, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> from !d2m.cb<memref<32x320xbf16, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %4 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %6 = d2m.wait %4 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %7 = d2m.reserve %5 : <memref<32x320xbf16, #l1>> -> memref<32x320xbf16, #l1>
        %8 = "d2m.tile_untilize_block"(%6, %7) : (memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, memref<32x320xbf16, #l1>) -> memref<32x320xbf16, #l1>
        d2m.pop %4 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.push %5 : <memref<32x320xbf16, #l1>>
      }
    }
    %alloc_18 = memref.alloc() : memref<3200x3200xbf16>
    d2m.to_host %alloc_12, %alloc_18 layout = #layout : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into memref<3200x3200xbf16>
    return %alloc_18 : memref<3200x3200xbf16>
  }
}
