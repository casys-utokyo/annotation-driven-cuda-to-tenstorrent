#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 3200x3200, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0)[s0, s1] -> (d0 floordiv 10 + s0 + (s1 floordiv 10) * 10)>
#map1 = affine_map<(d0)[s0] -> (d0 floordiv 10 + (s0 floordiv 10) * 10)>
#map2 = affine_map<(d0) -> (d0 mod 10)>
#map3 = affine_map<(d0)[s0] -> (d0 floordiv 10 + s0 + (s0 floordiv 10) * 10)>
module {
  func.func @_Z11syrk_kerneliiPfS__driver(%arg0: memref<3200x3200xbf16>, %arg1: memref<3200x3200xbf16>) -> memref<3200x3200xbf16> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 450560 : i64, tt.l1_persistent_base = 40960 : i64, tt.l1_pool = 40960 : i64} {
    %alloc = memref.alloc() {address = 152064 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 356864 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 115200 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 113152 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 117248 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 119296 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_6 = memref.alloc() {address = 8388608 : i64, alignment = 16 : i64} : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_7 = memref.alloc() {address = 28868608 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>
    d2m.to_device %arg0, %alloc_6 layout = #layout : memref<3200x3200xbf16> into memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_8 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_9 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_6 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        outs(%alloc_7 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        additionalArgs(%alloc_8, %alloc_9 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %0 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %1 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg2 = %c0 to %c10 step %c1 {
        %2 = arith.muli %arg2, %c32 : index
        d2m.remote_load %alloc_6[%core0, %core1] into %0 srcSlice[%2, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into !d2m.cb<memref<32x320xbf16, #l1>>
        d2m.remote_store %alloc_7[%core0, %core1] from %1 srcSlice[%arg2, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %1 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg2 = %c0 to %c10 step %c1 {
        %2 = d2m.wait %0 : <memref<32x320xbf16, #l1>> -> memref<32x320xbf16, #l1>
        %3 = d2m.reserve %1 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %4 = "d2m.tile_tilize_block"(%2, %3) : (memref<32x320xbf16, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        d2m.pop %0 : <memref<32x320xbf16, #l1>>
        d2m.push %1 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }
    %alloc_10 = memref.alloc() {address = 49348608 : i64, alignment = 16 : i64} : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_11 = memref.alloc() {address = 69828608 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>
    d2m.to_device %arg1, %alloc_10 layout = #layout : memref<3200x3200xbf16> into memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>
    %alloc_12 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_13 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_10 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        outs(%alloc_11 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        additionalArgs(%alloc_12, %alloc_13 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %0 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %1 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg2 = %c0 to %c10 step %c1 {
        %2 = arith.muli %arg2, %c32 : index
        d2m.remote_load %alloc_10[%core0, %core1] into %0 srcSlice[%2, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into !d2m.cb<memref<32x320xbf16, #l1>>
        d2m.remote_store %alloc_11[%core0, %core1] from %1 srcSlice[%arg2, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %1 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg2 = %c0 to %c10 step %c1 {
        %2 = d2m.wait %0 : <memref<32x320xbf16, #l1>> -> memref<32x320xbf16, #l1>
        %3 = d2m.reserve %1 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %4 = "d2m.tile_tilize_block"(%2, %3) : (memref<32x320xbf16, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #l1>) -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        d2m.pop %0 : <memref<32x320xbf16, #l1>>
        d2m.push %1 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_7 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %0 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_load %alloc_7[%core0, %core1] into %0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
    ^compute0:
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_11 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc_0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %0 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_load %alloc_11[%core0, %core1] into %0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
    ^compute0:
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc, %alloc_0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>, memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc_0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        additionalArgs(%alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>)
        {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %c100 = arith.constant 100 : index
      %0 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %1 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      scf.for %arg2 = %c0 to %c10 step %c1 {
        scf.for %arg3 = %c0 to %c10 step %c1 {
          scf.for %arg4 = %c0 to %c100 step %c1 {
            %2 = affine.apply #map(%arg3)[%core1, %core0]
            %3 = affine.apply #map1(%arg4)[%core1]
            %4 = affine.apply #map2(%arg3)
            %5 = affine.apply #map2(%arg4)
            d2m.remote_load %alloc[%2, %3] into %0 mcore[%c0, %core1] mshape[%c10, %c1] srcSlice[%4, %5] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            %6 = affine.apply #map3(%arg2)[%core0]
            %7 = affine.apply #map2(%arg2)
            d2m.remote_load %alloc[%6, %3] into %1 mcore[%core0, %c0] mshape[%c1, %c10] srcSlice[%7, %5] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
          } {d2m.blocking_loop = 1 : i64}
        }
      }
    }, {
    ^datamovement1:
    }, {
      %cst = arith.constant 3.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %cst_17 = arith.constant 2.000000e+00 : bf16
      %c100 = arith.constant 100 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      %0 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %1 = d2m.get_cb(4) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %2 = d2m.get_cb(5) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(6) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %4 = d2m.get_cb(7) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.reserve %3 {pin_cb = 6 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      %7 = d2m.tile_fill(%cst) : bf16 -> <32x32, bf16>
      memref.store %7, %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      %8 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      memref.store %8, %6[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %3 {pin_cb = 6 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %9 = d2m.wait %3 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %10 = d2m.reserve %4 {pin_cb = 7 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %dst_18 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      %11 = d2m.tile_fill(%cst_17) : bf16 -> <32x32, bf16>
      memref.store %11, %dst_18[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      %12 = memref.load %dst_18[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      memref.store %12, %10[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %4 {pin_cb = 7 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %13 = d2m.wait %4 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %14 = d2m.reserve %0 {pin_cb = 3 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %15 = d2m.reserve %5 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      scf.for %arg2 = %c0 to %c10 step %c1 {
        scf.for %arg3 = %c0 to %c10 step %c1 {
          %16 = affine.apply #map2(%arg2)
          %17 = affine.apply #map2(%arg3)
          %18 = memref.load %15[%16, %17] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          %dst_19 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
          %cast = memref.cast %14 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1> to memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1>
          scf.for %arg4 = %c0 to %c100 step %c1 {
            %28 = d2m.wait %1 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %29 = d2m.wait %2 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %cast_20 = memref.cast %29 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1> to memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1>
            %cast_21 = memref.cast %28 : memref<1x1x!ttcore.tile<32x32, bf16>, #l1> to memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1>
            "d2m.tile_matmul_block"(%cast_20, %cast_21, %cast) <{transpose_b = true}> : (memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1>) -> ()
            d2m.pop %2 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
            d2m.pop %1 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
          } {d2m.blocking_loop = 1 : i64}
          %19 = memref.load %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
          %20 = memref.load %9[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
          %21 = "d2m.tile_mul"(%18, %20) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          memref.store %21, %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
          %22 = memref.load %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
          %23 = memref.load %13[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
          %24 = "d2m.tile_mul"(%19, %23) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          memref.store %24, %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
          %25 = memref.load %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
          %26 = "d2m.tile_add"(%22, %25) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          memref.store %26, %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
          %27 = memref.load %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
          memref.store %27, %15[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        }
      }
      d2m.push %5 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.pop %4 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.pop %3 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
    }
    %alloc_14 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_15 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc_11 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %0 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.remote_store %alloc_11[%core0, %core1] from %0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }, {
      %0 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %1 = d2m.reserve %0 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %0 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
    }
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_11 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        outs(%alloc_10 : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram>)
        additionalArgs(%alloc_14, %alloc_15 : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>, memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>)
        {
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %0 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      %1 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      scf.for %arg2 = %c0 to %c10 step %c1 {
        %2 = arith.muli %arg2, %c32 : index
        d2m.remote_load %alloc_11[%core0, %core1] into %1 srcSlice[%arg2, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_store %alloc_10[%core0, %core1] from %0 srcSlice[%2, %c0] : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> from !d2m.cb<memref<32x320xbf16, #l1>>
      }
    }, {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0 = d2m.get_cb(3) resolution_stage =  compile : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
      %1 = d2m.get_cb(2) resolution_stage =  compile : <memref<32x320xbf16, #l1>>
      scf.for %arg2 = %c0 to %c10 step %c1 {
        %2 = d2m.wait %0 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x10x!ttcore.tile<32x32, bf16>, #l1>
        %3 = d2m.reserve %1 : <memref<32x320xbf16, #l1>> -> memref<32x320xbf16, #l1>
        %4 = "d2m.tile_untilize_block"(%2, %3) : (memref<1x10x!ttcore.tile<32x32, bf16>, #l1>, memref<32x320xbf16, #l1>) -> memref<32x320xbf16, #l1>
        d2m.pop %0 : <memref<1x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.push %1 : <memref<32x320xbf16, #l1>>
      }
    }
    %alloc_16 = memref.alloc() : memref<3200x3200xbf16>
    d2m.to_host %alloc_10, %alloc_16 layout = #layout : memref<10x10x320x320xbf16, #ttcore.shard<640x2, 1>, #dram> into memref<3200x3200xbf16>
    return %alloc_16 : memref<3200x3200xbf16>
  }
}
