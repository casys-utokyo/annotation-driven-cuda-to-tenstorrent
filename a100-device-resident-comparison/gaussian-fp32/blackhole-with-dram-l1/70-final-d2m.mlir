#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 5120x5120, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0)[s0] -> ((d0 floordiv 32) floordiv 16 + (s0 floordiv 10) * 10)>
#map1 = affine_map<(d0) -> ((d0 floordiv 32) mod 16)>
#map2 = affine_map<(d0) -> (d0 mod 32)>
#map3 = affine_map<()[s0] -> (s0 + (s0 floordiv 10) * 10)>
#map4 = affine_map<(d0, d1, d2)[s0] -> (d0 - d1 + d2 * 32 + s0 * 512)>
#map5 = affine_map<(d0, d1, d2)[s0] -> (d0 - d1 + d2 * 32 + s0 * 512 - 1)>
#map6 = affine_map<(d0) -> (d0 mod 16)>
module {
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xf32>) -> memref<5120x5120xf32> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 1261568 : i64, tt.l1_persistent_base = 212992 : i64, tt.l1_pool = 212992 : i64} {
    %alloc = memref.alloc() {address = 324096 : i64, alignment = 16 : i64} : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 115200 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 119296 : i64, alignment = 16 : i64} : memref<16x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 184832 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 250368 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 315904 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_6 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram>
    %alloc_7 = memref.alloc() {address = 113246208 : i64, alignment = 16 : i64} : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>
    d2m.to_device %arg0, %alloc_6 layout = #layout : memref<5120x5120xf32> into memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram>
    %alloc_8 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x512xf32, #ttcore.cb_layout<2048x4, 1>, #l1>
    %alloc_9 = memref.alloc() {address = 176640 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_6 : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram>)
        outs(%alloc_7 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>)
        additionalArgs(%alloc_8, %alloc_9 : memref<32x512xf32, #ttcore.cb_layout<2048x4, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>)
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
        d2m.remote_load %alloc_6[%core0, %core1] into %2 srcSlice[%4, %c0] : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram> into !d2m.cb<memref<32x512xf32, #l1>>
        d2m.remote_store %alloc_7[%core0, %core1] from %3 srcSlice[%arg1, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram> from !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
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
    // FOLDED:
    // The landing and writeback generics are dissolved into the workload
    // generic's DM0, bracketing its epoch loop.
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
        outs(%alloc, %alloc_7 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>, memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>)
        additionalArgs(%alloc_0, %alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %0, %1 : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<16x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %cst = arith.constant 1.000000e+00 : f32
      %c16 = arith.constant 16 : index
      %cst_13 = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c5119 = arith.constant 5119 : index
      %2 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %4 = d2m.get_cb(5) resolution_stage =  compile : <memref<16x1x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %6 = d2m.get_cb(7) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %7 = d2m.get_cb(8) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %8 = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %fold_cb0 = d2m.get_cb(0) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      d2m.remote_load %alloc_7[%core0, %core1] into %fold_cb0 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram> into !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c5119 step %c1 {
        %9 = affine.apply #map(%arg1)[%core0]
        %10 = affine.apply #map(%arg1)[%core1]
        %11 = affine.apply #map1(%arg1)
        %12 = affine.apply #map2(%arg1)
        %13 = d2m.wait %8 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
        %14 = d2m.get_arg(9) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %14, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %14, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc[%9, %10] into %2 mcore[%c0, %c0] mshape[%c10, %c10] srcSlice[%11, %11] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        %15 = d2m.wait %2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %16 = d2m.tile_elem_load %2[%c0, %12, %12] : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, index, index, index -> i32
        %17 = arith.bitcast %16 : i32 to f32
        d2m.pop %2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        %18 = arith.divf %cst, %17 : f32
        %19 = arith.bitcast %18 : f32 to i32
        d2m.tile_elem_store %19, %3[%c0, %12, %12] : i32, !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, index, index, index
        %20 = affine.apply #map3()[%core0]
        d2m.remote_load %alloc[%20, %10] into %4 mcore[%core0, %c0] mshape[%c1, %c10] srcSlice[%c0, %11] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> into !d2m.cb<memref<16x1x!ttcore.tile<32x32, f32>, #l1>>
        %21 = affine.apply #map3()[%core1]
        d2m.remote_load %alloc[%9, %21] into %5 mcore[%c0, %core1] mshape[%c10, %c1] srcSlice[%11, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> into !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        %22 = d2m.get_arg(10) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %22, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %22, %c100 reset %c0 : !d2m.local_semaphore
        d2m.pop %8 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
        %23 = d2m.wait %5 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        %24 = d2m.reserve %6 {pin_cb = 6 : i64} : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          scf.for %arg3 = %c0 to %c32 step %c1 {
            %26 = d2m.tile_elem_load %5[%arg2, %12, %arg3] : !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>, index, index, index -> i32
            %27 = arith.bitcast %26 : i32 to f32
            %28 = affine.apply #map4(%arg3, %arg1, %arg2)[%core1]
            %29 = arith.cmpi sge, %28, %c0 : index
            %30 = arith.select %29, %27, %cst_13 : f32
            %31 = arith.bitcast %30 : f32 to i32
            d2m.tile_elem_store %31, %6[%arg2, %c0, %arg3] : i32, !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>, index, index, index
          }
        }
        d2m.push %6 {pin_cb = 6 : i64} : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %5 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        %25 = d2m.wait %4 : <memref<16x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x1x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          %26 = d2m.reserve %7 {pin_cb = 7 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          scf.for %arg3 = %c0 to %c32 step %c1 {
            %27 = d2m.tile_elem_load %4[%arg2, %arg3, %12] : !d2m.cb<memref<16x1x!ttcore.tile<32x32, f32>, #l1>>, index, index, index -> i32
            %28 = arith.bitcast %27 : i32 to f32
            %29 = affine.apply #map5(%arg3, %arg1, %arg2)[%core0]
            %30 = arith.cmpi sge, %29, %c0 : index
            %31 = arith.select %30, %28, %cst_13 : f32
            %32 = d2m.tile_elem_load %3[%c0, %12, %12] : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, index, index, index -> i32
            %33 = arith.bitcast %32 : i32 to f32
            %34 = arith.mulf %31, %33 : f32
            %35 = arith.bitcast %34 : f32 to i32
            d2m.tile_elem_store %35, %7[%c0, %arg3, %12] : i32, !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, index, index, index
          }
          d2m.bcast_col_to_tile(%26, %12) to %26 at %c0, %c0 of %c1 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index, index
          d2m.push %7 {pin_cb = 7 : i64} : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        }
        d2m.pop %4 : <memref<16x1x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 5119 : i64}
      %fold_store = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      d2m.remote_store %alloc_7[%core0, %core1] from %fold_store : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram> from !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      d2m.atomic_barrier
    }, {
    ^datamovement1:
    }, {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c5119 = arith.constant 5119 : index
      %2 = d2m.get_cb(7) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.get_cb(8) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %4 = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      %5 = d2m.reserve %4 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      d2m.push %4 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %arg1 = %c0 to %c5119 step %c1 {
        %6 = d2m.reserve %4 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
        %7 = d2m.wait %2 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x16x!ttcore.tile<32x32, f32>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          %8 = d2m.wait %3 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          scf.for %arg3 = %c0 to %c16 step %c1 {
            %9 = affine.apply #map6(%arg2)
            %10 = affine.apply #map6(%arg3)
            %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %11 = memref.load %6[%9, %10] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
            memref.store %11, %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %12 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %13 = memref.load %7[%c0, %arg3] : memref<1x16x!ttcore.tile<32x32, f32>, #l1>
            memref.store %13, %dst[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %14 = memref.load %dst[%c1] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %15 = memref.load %8[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            memref.store %15, %dst[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %16 = memref.load %dst[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %17 = "d2m.tile_bcast_binary"(%16, %14) <{bcast_type = #d2m<tile_bcast_type row>, binop_type = #d2m<tile_bcast_binary_op_type mul>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %17, %dst[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %18 = memref.load %dst[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %19 = "d2m.tile_sub"(%12, %18) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            memref.store %19, %dst[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            %20 = memref.load %dst[%c2] : memref<16x!ttcore.tile<32x32, f32>, #dst>
            memref.store %20, %6[%arg2, %arg3] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
          }
          d2m.pop %3 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
        }
        d2m.pop %2 : <memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        d2m.push %4 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.sync_loop = 5119 : i64}
    }
    %alloc_10 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x512xf32, #ttcore.cb_layout<2048x4, 1>, #l1>
    %alloc_11 = memref.alloc() {address = 176640 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_7 : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram>)
        outs(%alloc_6 : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram>)
        additionalArgs(%alloc_10, %alloc_11 : memref<32x512xf32, #ttcore.cb_layout<2048x4, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 1>, #l1>)
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
        d2m.remote_load %alloc_7[%core0, %core1] into %3 srcSlice[%arg1, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #dram> into !d2m.cb<memref<1x16x!ttcore.tile<32x32, f32>, #l1>>
        d2m.remote_store %alloc_6[%core0, %core1] from %2 srcSlice[%4, %c0] : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram> from !d2m.cb<memref<32x512xf32, #l1>>
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
    %alloc_12 = memref.alloc() : memref<5120x5120xf32>
    d2m.to_host %alloc_6, %alloc_12 layout = #layout : memref<10x10x512x512xf32, #ttcore.shard<2048x4, 1>, #dram> into memref<5120x5120xf32>
    return %alloc_12 : memref<5120x5120xf32>
  }
}
