#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 5120x5120, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0)[s0] -> ((d0 floordiv 32) floordiv 16 + (s0 floordiv 10) * 10)>
#map1 = affine_map<(d0) -> ((d0 floordiv 32) mod 16)>
#map2 = affine_map<(d0) -> (d0 mod 32)>
#map3 = affine_map<()[s0] -> (s0 + (s0 floordiv 10) * 10)>
#map4 = affine_map<(d0, d1)[s0] -> (-d0 + d1 * 32 + s0 * 512)>
#map5 = affine_map<(d0, d1)[s0] -> (-d0 + d1 * 32 + s0 * 512 - 1)>
#map6 = affine_map<(d0) -> (d0 mod 16)>
module {
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xf32>) -> memref<5120x5120xf32> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 1343488 : i64, tt.l1_persistent_base = 294912 : i64, tt.l1_pool = 294912 : i64} {
    %alloc = memref.alloc() {address = 406016 : i64, alignment = 16 : i64} : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 115200 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 119296 : i64, alignment = 16 : i64} : memref<16x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 184832 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 250368 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 315904 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    %alloc_6 = memref.alloc() {address = 381440 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_7 = memref.alloc() {address = 389632 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_8 = memref.alloc() {address = 397824 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_9 = memref.alloc() {address = 401920 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_10 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram>
    %alloc_11 = memref.alloc() {address = 113246208 : i64, alignment = 16 : i64} : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>
    d2m.to_device %arg0, %alloc_10 layout = #layout : memref<5120x5120xf32> into memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram>
    %alloc_12 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x512xf32, #ttcore.cb_layout<2048x4, 1>, #l1>
    %alloc_13 = memref.alloc() {address = 176640 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_10 : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram>)
        outs(%alloc_11 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>)
        additionalArgs(%alloc_12, %alloc_13 : memref<32x512xf32, #ttcore.cb_layout<2048x4, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x512xf32, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_10[%core0, %core1] into %2 srcSlice[%4, %c0] : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram> into !d2m.cb<memref<32x512xf32, #l1>>
        d2m.remote_store %alloc_11[%core0, %core1] from %3 srcSlice[%arg1, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram> from !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x512xf32, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = d2m.wait %2 : <memref<32x512xf32, #l1>> -> memref<32x512xf32, #l1>
        %5 = d2m.reserve %3 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        %6 = "d2m.tile_tilize_block"(%4, %5) : (memref<32x512xf32, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #l1>) -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        d2m.pop %2 : <memref<32x512xf32, #l1>>
        d2m.push %3 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      }
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_11 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>)
        outs(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      d2m.remote_load %alloc_11[%core0, %core1] into %2 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram> into !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0:
    }
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
        outs(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
        additionalArgs(%alloc_0, %alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %alloc_6, %alloc_7, %alloc_8, %alloc_9, %0, %1 : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<16x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c16 = arith.constant 16 : index
      %c5119 = arith.constant 5119 : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %4 = d2m.get_cb(4) resolution_stage =  compile : <memref<16x1x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %6 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %7 = d2m.get_cb(7) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %8 = d2m.get_cb(8) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %9 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %10 = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      scf.for %arg1 = %c0 to %c5119 step %c1 {
        %11 = affine.apply #map(%arg1)[%core0]
        %12 = affine.apply #map(%arg1)[%core1]
        %13 = affine.apply #map1(%arg1)
        %14 = affine.apply #map2(%arg1)
        %15 = d2m.wait %10 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
        %16 = d2m.get_arg(12) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %16, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %16, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc[%11, %12] into %2 mcore[%c0, %c0] mshape[%c10, %c10] srcSlice[%13, %13] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        %17 = d2m.wait %2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %18 = d2m.reserve %3 {pin_cb = 3 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        d2m.bcast_cell_to_tile(%17, %14, %14) to %18 at %c0, %c0 of %c1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
        d2m.push %3 {pin_cb = 3 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        %19 = affine.apply #map3()[%core0]
        d2m.remote_load %alloc[%19, %12] into %4 mcore[%core0, %c0] mshape[%c1, %c10] srcSlice[%c0, %13] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> into !d2m.cb<memref<16x1x!ttcore.tile<32x32, f32>, #l1>>
        %20 = affine.apply #map3()[%core1]
        d2m.remote_load %alloc[%11, %20] into %5 mcore[%c0, %core1] mshape[%c10, %c1] srcSlice[%13, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> into !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        %21 = d2m.get_arg(13) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %21, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %21, %c100 reset %c0 : !d2m.local_semaphore
        d2m.pop %10 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
        %22 = d2m.wait %5 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        %23 = d2m.reserve %6 {pin_cb = 6 : i64} : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          d2m.bcast_row_to_tile(%22, %14) to %23 at %arg2, %arg2 of %c16 : memref<1x16x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x16x!ttcore.tile<32x32, f32>, #l1>, index, index, index
        }
        d2m.push %6 {pin_cb = 6 : i64} : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %5 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        %24 = d2m.reserve %7 {pin_cb = 7 : i64} : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          %26 = affine.apply #map4(%arg1, %arg2)[%core1]
          %27 = arith.subi %c0, %26 : index
          %28 = arith.maxsi %27, %c0 : index
          %29 = arith.shli %c1, %28 : index
          %30 = arith.subi %29, %c1 : index
          d2m.write_col_mask_tile(%30) to %7 at %arg2 of %c16 : index, !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>, index, index
        }
        d2m.push %7 {pin_cb = 7 : i64} : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        %25 = d2m.wait %4 : <memref<16x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x1x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          %26 = d2m.reserve %8 {pin_cb = 8 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          d2m.bcast_col_to_tile(%25, %14) to %26 at %arg2, %c0 of %c1 : memref<16x1x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
          d2m.push %8 {pin_cb = 8 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          %27 = affine.apply #map5(%arg1, %arg2)[%core0]
          %28 = d2m.reserve %9 {pin_cb = 9 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          %29 = arith.subi %c0, %27 : index
          %30 = arith.maxsi %29, %c0 : index
          %31 = arith.shli %c1, %30 : index
          %32 = arith.subi %31, %c1 : index
          d2m.write_row_mask_tile(%32) to %9 at %c0 of %c1 : index, !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, index, index
          d2m.push %9 {pin_cb = 9 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        }
        d2m.pop %4 : <memref<16x1x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 5119 : i64}
      d2m.atomic_barrier
    }, {
    ^datamovement1:
    }, {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c5119 = arith.constant 5119 : index
      %2 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %4 = d2m.get_cb(7) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(8) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %6 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %7 = d2m.get_cb(10) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %8 = d2m.get_cb(11) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %9 = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      %10 = d2m.reserve %9 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      d2m.push %9 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c5119 step %c1 {
        %11 = d2m.reserve %7 {pin_cb = 10 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
        %12 = d2m.wait %2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %13 = memref.load %12[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        memref.store %13, %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
        %14 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
        %15 = "d2m.tile_recip"(%14) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        memref.store %15, %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
        %16 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
        d2m.pop %2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        memref.store %16, %11[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        d2m.push %7 {pin_cb = 10 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        %17 = d2m.reserve %9 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
        %18 = d2m.wait %3 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        %19 = d2m.wait %4 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        %20 = d2m.wait %7 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          %21 = d2m.reserve %8 {pin_cb = 11 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          %dst_17 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
          %22 = d2m.wait %5 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          %23 = memref.load %22[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          memref.store %23, %dst_17[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
          %24 = memref.load %dst_17[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
          %25 = memref.load %20[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          memref.store %25, %dst_17[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
          %26 = memref.load %dst_17[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
          %27 = "d2m.tile_mul"(%24, %26) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          memref.store %27, %dst_17[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
          %28 = memref.load %dst_17[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
          d2m.pop %5 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          memref.store %28, %21[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          d2m.push %8 {pin_cb = 11 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          %29 = d2m.wait %6 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          %30 = d2m.wait %8 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          scf.for %arg3 = %c0 to %c16 step %c1 {
            %dst_18 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %31 = memref.load %18[%c0, %arg3] : memref<1x16x!ttcore.tile<32x32, f32>, #l1>
            memref.store %31, %dst_18[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %32 = memref.load %dst_18[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %33 = affine.apply #map6(%arg2)
            %34 = affine.apply #map6(%arg3)
            %35 = memref.load %17[%33, %34] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
            memref.store %35, %dst_18[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %36 = memref.load %dst_18[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %37 = memref.load %30[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %37, %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %38 = memref.load %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %39 = "d2m.tile_mul"(%38, %32) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %39, %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %40 = memref.load %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %41 = "d2m.tile_sub"(%36, %40) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %41, %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %42 = memref.load %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %43 = memref.load %29[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %43, %dst_18[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %44 = memref.load %dst_18[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %45 = "d2m.tile_where"(%44, %36, %42) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %45, %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %46 = memref.load %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %47 = memref.load %19[%c0, %arg3] : memref<1x16x!ttcore.tile<32x32, f32>, #l1>
            memref.store %47, %dst_18[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %48 = memref.load %dst_18[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %49 = "d2m.tile_where"(%48, %36, %46) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %49, %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %50 = memref.load %dst_18[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            memref.store %50, %17[%arg2, %arg3] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
          }
          d2m.pop %8 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
          d2m.pop %6 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        }
        d2m.pop %7 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %4 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %3 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        d2m.push %9 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 5119 : i64}
    }
    %alloc_14 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x512xf32, #ttcore.cb_layout<2048x4, 1>, #l1>
    %alloc_15 = memref.alloc() {address = 176640 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
        outs(%alloc_11 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      d2m.remote_store %alloc_11[%core0, %core1] from %2 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram> from !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
    }, {
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.reserve %2 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      d2m.push %2 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_11 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>)
        outs(%alloc_10 : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram>)
        additionalArgs(%alloc_14, %alloc_15 : memref<32x512xf32, #ttcore.cb_layout<2048x4, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x512xf32, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_11[%core0, %core1] into %3 srcSlice[%arg1, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram> into !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_store %alloc_10[%core0, %core1] from %2 srcSlice[%4, %c0] : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram> from !d2m.cb<memref<32x512xf32, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %2 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x512xf32, #l1>>
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = d2m.wait %2 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        %5 = d2m.reserve %3 : <memref<32x512xf32, #l1>> -> memref<32x512xf32, #l1>
        %6 = "d2m.tile_untilize_block"(%4, %5) : (memref<1x16x!ttcore.tile<32x32, f32>, #l1>, memref<32x512xf32, #l1>) -> memref<32x512xf32, #l1>
        d2m.pop %2 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        d2m.push %3 : <memref<32x512xf32, #l1>>
      }
    }
    %alloc_16 = memref.alloc() : memref<5120x5120xf32>
    d2m.to_host %alloc_10, %alloc_16 layout = #layout : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram> into memref<5120x5120xf32>
    return %alloc_16 : memref<5120x5120xf32>
  }
}
