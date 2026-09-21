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
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xbf16>) -> memref<5120x5120xbf16> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 630784 : i64, tt.l1_persistent_base = 106496 : i64, tt.l1_pool = 106496 : i64} {
    %alloc = memref.alloc() {address = 217600 : i64, alignment = 16 : i64} : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 113152 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 115200 : i64, alignment = 16 : i64} : memref<16x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 147968 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<32768x2048, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 180736 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<32768x2048, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 213504 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
    %alloc_6 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x512x512xbf16, #ttcore.shard<1024x2, 1>, #dram>
    %alloc_7 = memref.alloc() {address = 60817408 : i64, alignment = 16 : i64} : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram>
    d2m.to_device %arg0, %alloc_6 layout = #layout : memref<5120x5120xbf16> into memref<10x10x512x512xbf16, #ttcore.shard<1024x2, 1>, #dram>
    %alloc_8 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x512xbf16, #ttcore.cb_layout<1024x2, 1>, #l1>
    %alloc_9 = memref.alloc() {address = 143872 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<32768x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_6 : memref<10x10x512x512xbf16, #ttcore.shard<1024x2, 1>, #dram>)
        outs(%alloc_7 : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram>)
        additionalArgs(%alloc_8, %alloc_9 : memref<32x512xbf16, #ttcore.cb_layout<1024x2, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<32768x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x512xbf16, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_6[%core0, %core1] into %2 srcSlice[%4, %c0] : memref<10x10x512x512xbf16, #ttcore.shard<1024x2, 1>, #dram> into !d2m.cb<memref<32x512xbf16, #l1>>
        d2m.remote_store %alloc_7[%core0, %core1] from %3 srcSlice[%arg1, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram> from !d2m.cb<memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x512xbf16, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = d2m.wait %2 : <memref<32x512xbf16, #l1>> -> memref<32x512xbf16, #l1>
        %5 = d2m.reserve %3 : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x16x!ttcore.tile<32x32, bf16>, #l1>
        %6 = "d2m.tile_tilize_block"(%4, %5) : (memref<32x512xbf16, #l1>, memref<1x16x!ttcore.tile<32x32, bf16>, #l1>) -> memref<1x16x!ttcore.tile<32x32, bf16>, #l1>
        d2m.pop %2 : <memref<32x512xbf16, #l1>>
        d2m.push %3 : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_7 : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram>)
        outs(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_load %alloc_7[%core0, %core1] into %2 : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram> into !d2m.cb<memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
    ^compute0:
    }
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #l1>)
        outs(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #l1>)
        additionalArgs(%alloc_0, %alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %0, %1 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<16x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<32768x2048, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<32768x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c100 = arith.constant 100 : index
      %c16_i32 = arith.constant 16 : i32
      %cst = arith.constant 1.000000e+00 : f32
      %c16 = arith.constant 16 : index
      %cst_13 = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c5119 = arith.constant 5119 : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %4 = d2m.get_cb(4) resolution_stage =  compile : <memref<16x1x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      %7 = d2m.get_cb(7) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %8 = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      scf.for %arg1 = %c0 to %c5119 step %c1 {
        %9 = affine.apply #map(%arg1)[%core0]
        %10 = affine.apply #map(%arg1)[%core1]
        %11 = affine.apply #map1(%arg1)
        %12 = affine.apply #map2(%arg1)
        %13 = d2m.wait %8 : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<16x16x!ttcore.tile<32x32, bf16>, #l1>
        %14 = d2m.get_arg(8) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %14, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %14, %c100 reset %c0 : !d2m.local_semaphore
        d2m.remote_load %alloc[%9, %10] into %2 mcore[%c0, %c0] mshape[%c10, %c10] srcSlice[%11, %11] : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #l1> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        %15 = d2m.wait %2 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
        %16 = d2m.tile_elem_load %2[%c0, %12, %12] : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index, index -> i16
        d2m.pop %2 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        %17 = arith.extui %16 : i16 to i32
        %18 = arith.shli %17, %c16_i32 : i32
        %19 = arith.bitcast %18 : i32 to f32
        %20 = arith.divf %cst, %19 : f32
        %21 = arith.bitcast %20 : f32 to i32
        %22 = arith.shrui %21, %c16_i32 : i32
        %23 = arith.trunci %22 : i32 to i16
        d2m.tile_elem_store %23, %3[%c0, %12, %12] : i16, !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index, index
        %24 = affine.apply #map3()[%core0]
        d2m.remote_load %alloc[%24, %10] into %4 mcore[%core0, %c0] mshape[%c1, %c10] srcSlice[%c0, %11] : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #l1> into !d2m.cb<memref<16x1x!ttcore.tile<32x32, bf16>, #l1>>
        %25 = affine.apply #map3()[%core1]
        d2m.remote_load %alloc[%9, %25] into %5 mcore[%c0, %core1] mshape[%c10, %c1] srcSlice[%11, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #l1> into !d2m.cb<memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
        %26 = d2m.get_arg(9) : !d2m.local_semaphore
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            d2m.semaphore_inc %26, %c1, core[%arg2, %arg3] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %26, %c100 reset %c0 : !d2m.local_semaphore
        d2m.pop %8 : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
        %27 = d2m.wait %5 : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x16x!ttcore.tile<32x32, bf16>, #l1>
        %28 = d2m.reserve %6 {pin_cb = 6 : i64} : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x16x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          scf.for %arg3 = %c0 to %c32 step %c1 {
            %30 = d2m.tile_elem_load %5[%arg2, %12, %arg3] : !d2m.cb<memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>, index, index, index -> i16
            %31 = arith.bitcast %30 : i16 to bf16
            %32 = affine.apply #map4(%arg3, %arg1, %arg2)[%core1]
            %33 = arith.cmpi sge, %32, %c0 : index
            %34 = arith.select %33, %31, %cst_13 : bf16
            %35 = arith.bitcast %34 : bf16 to i16
            d2m.tile_elem_store %35, %6[%arg2, %c0, %arg3] : i16, !d2m.cb<memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>, index, index, index
          }
        }
        d2m.push %6 {pin_cb = 6 : i64} : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.pop %5 : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
        %29 = d2m.wait %4 : <memref<16x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<16x1x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          %30 = d2m.reserve %7 {pin_cb = 7 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
          scf.for %arg3 = %c0 to %c32 step %c1 {
            %31 = d2m.tile_elem_load %4[%arg2, %arg3, %12] : !d2m.cb<memref<16x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index, index -> i16
            %32 = arith.bitcast %31 : i16 to bf16
            %33 = affine.apply #map5(%arg3, %arg1, %arg2)[%core0]
            %34 = arith.cmpi sge, %33, %c0 : index
            %35 = arith.select %34, %32, %cst_13 : bf16
            %36 = d2m.tile_elem_load %3[%c0, %12, %12] : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index, index -> i16
            %37 = arith.bitcast %35 : bf16 to i16
            %38 = arith.extui %37 : i16 to i32
            %39 = arith.shli %38, %c16_i32 : i32
            %40 = arith.bitcast %39 : i32 to f32
            %41 = arith.extui %36 : i16 to i32
            %42 = arith.shli %41, %c16_i32 : i32
            %43 = arith.bitcast %42 : i32 to f32
            %44 = arith.mulf %40, %43 : f32
            %45 = arith.bitcast %44 : f32 to i32
            %46 = arith.shrui %45, %c16_i32 : i32
            %47 = arith.trunci %46 : i32 to i16
            d2m.tile_elem_store %47, %7[%c0, %arg3, %12] : i16, !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>, index, index, index
          }
          d2m.bcast_col_to_tile(%30, %12) to %30 at %c0, %c0 of %c1 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, index, index, index
          d2m.push %7 {pin_cb = 7 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        }
        d2m.pop %4 : <memref<16x1x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 5119 : i64}
      d2m.atomic_barrier
    }, {
    ^datamovement1:
    }, {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c5119 = arith.constant 5119 : index
      %2 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(7) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %4 = d2m.get_cb(1) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.reserve %4 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<16x16x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %4 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c5119 step %c1 {
        %6 = d2m.reserve %4 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<16x16x!ttcore.tile<32x32, bf16>, #l1>
        %7 = d2m.wait %2 : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x16x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c16 step %c1 {
          %8 = d2m.wait %3 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
          scf.for %arg3 = %c0 to %c16 step %c1 {
            %9 = affine.apply #map6(%arg2)
            %10 = affine.apply #map6(%arg3)
            %11 = memref.load %6[%9, %10] : memref<16x16x!ttcore.tile<32x32, bf16>, #l1>
            %12 = memref.load %7[%c0, %arg3] : memref<1x16x!ttcore.tile<32x32, bf16>, #l1>
            %13 = memref.load %8[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %14 = "d2m.tile_bcast_binary"(%13, %12) <{bcast_type = #d2m<tile_bcast_type row>, binop_type = #d2m<tile_bcast_binary_op_type mul>}> : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %14, %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %15 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %16 = "d2m.tile_sub"(%11, %15) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %16, %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %17 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            memref.store %17, %6[%arg2, %arg3] : memref<16x16x!ttcore.tile<32x32, bf16>, #l1>
          }
          d2m.pop %3 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
        }
        d2m.pop %2 : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.push %4 {pin_cb = 1 : i64} : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
      } {d2m.sync_loop = 5119 : i64}
    }
    %alloc_10 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x512xbf16, #ttcore.cb_layout<1024x2, 1>, #l1>
    %alloc_11 = memref.alloc() {address = 143872 : i64, alignment = 16 : i64} : memref<1x16x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<32768x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #l1>)
        outs(%alloc_7 : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_store %alloc_7[%core0, %core1] from %2 : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram> from !d2m.cb<memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
      %2 = d2m.get_cb(0) resolution_stage =  compile : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.reserve %2 : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<16x16x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %2 : <memref<16x16x!ttcore.tile<32x32, bf16>, #l1>>
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_7 : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram>)
        outs(%alloc_6 : memref<10x10x512x512xbf16, #ttcore.shard<1024x2, 1>, #dram>)
        additionalArgs(%alloc_10, %alloc_11 : memref<32x512xbf16, #ttcore.cb_layout<1024x2, 1>, #l1>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<32768x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x512xbf16, #l1>>
      %3 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = arith.muli %arg1, %c32 : index
        d2m.remote_load %alloc_7[%core0, %core1] into %3 srcSlice[%arg1, %c0] : memref<10x10x16x16x!ttcore.tile<32x32, bf16>, #ttcore.shard<32768x2048, 1>, #dram> into !d2m.cb<memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_store %alloc_6[%core0, %core1] from %2 srcSlice[%4, %c0] : memref<10x10x512x512xbf16, #ttcore.shard<1024x2, 1>, #dram> from !d2m.cb<memref<32x512xbf16, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %2 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x512xbf16, #l1>>
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = d2m.wait %2 : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x16x!ttcore.tile<32x32, bf16>, #l1>
        %5 = d2m.reserve %3 : <memref<32x512xbf16, #l1>> -> memref<32x512xbf16, #l1>
        %6 = "d2m.tile_untilize_block"(%4, %5) : (memref<1x16x!ttcore.tile<32x32, bf16>, #l1>, memref<32x512xbf16, #l1>) -> memref<32x512xbf16, #l1>
        d2m.pop %2 : <memref<1x16x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.push %3 : <memref<32x512xbf16, #l1>>
      }
    }
    %alloc_12 = memref.alloc() : memref<5120x5120xbf16>
    d2m.to_host %alloc_6, %alloc_12 layout = #layout : memref<10x10x512x512xbf16, #ttcore.shard<1024x2, 1>, #dram> into memref<5120x5120xbf16>
    return %alloc_12 : memref<5120x5120xbf16>
  }
}
