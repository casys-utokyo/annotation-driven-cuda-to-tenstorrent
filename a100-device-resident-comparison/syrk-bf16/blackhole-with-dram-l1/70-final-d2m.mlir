#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 3200x3200, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<()[s0, s1] -> (s0 + (s1 floordiv 10) * 10)>
#map1 = affine_map<()[s0] -> (s0 + (s0 floordiv 10) * 10)>
#map2 = affine_map<(d0)[s0] -> (d0 + (s0 floordiv 10) * 10)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0) -> (d0 mod 10)>
module {
  func.func @_Z11syrk_kerneliiPfS__driver(%arg0: memref<3200x3200xbf16>, %arg1: memref<3200x3200xbf16>) -> memref<3200x3200xbf16> attributes {tt.d2m_emittable = true, tt.d2m_unhandled = 0 : i64, tt.function_type = "forward_device", tt.l1_peak = 1437696 : i64, tt.l1_persistent_base = 1028096 : i64, tt.l1_pool = 1028096 : i64} {
    %alloc = memref.alloc() {address = 1139200 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 1344000 : i64, alignment = 16 : i64} : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 315904 : i64, alignment = 16 : i64} : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_3 = memref.alloc() {address = 725504 : i64, alignment = 16 : i64} : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>
    %alloc_4 = memref.alloc() {address = 1135104 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
    %alloc_5 = memref.alloc() {address = 1137152 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>
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
    // FOLDED:
    // The landing and writeback generics are dissolved into the workload
    // generic's DM0, bracketing its epoch loop.
    %fold_sem = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %rep_sem = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<10x10>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc, %alloc_0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>, memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>)
        outs(%alloc_0, %alloc_7, %alloc_11 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1>, memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>, memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram>)
        additionalArgs(%alloc_1, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %fold_sem, %rep_sem : memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 1>, #l1>, !d2m.local_semaphore, !d2m.local_semaphore)
        {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0 = d2m.get_cb(6) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %1 = d2m.get_cb(7) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %fold_cb0 = d2m.get_cb(0) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %fold_c0 = arith.constant 0 : index
      %fold_c1 = arith.constant 1 : index
      %fold_c10 = arith.constant 10 : index
      %fold_c100 = arith.constant 100 : index
      %fold_store = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %rep_cN = arith.constant 100 : index
      scf.for %rep = %fold_c0 to %rep_cN step %fold_c1 {
        d2m.remote_load %alloc_7[%core0, %core1] into %fold_cb0 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        %fold_cb1 = d2m.get_cb(1) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        d2m.remote_load %alloc_11[%core0, %core1] into %fold_cb1 : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        %fold_bar = d2m.get_arg(10) : !d2m.local_semaphore
        scf.for %fby = %fold_c0 to %fold_c10 step %fold_c1 {
          scf.for %fbx = %fold_c0 to %fold_c10 step %fold_c1 {
            d2m.semaphore_inc %fold_bar, %fold_c1, core[%fby, %fbx] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %fold_bar, %fold_c100 reset %fold_c0 : !d2m.local_semaphore
        %2 = affine.apply #map()[%core1, %core0]
        %3 = affine.apply #map1()[%core0]
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %4 = affine.apply #map2(%arg2)[%core1]
          d2m.remote_load %alloc[%2, %4] into %0 mcore[%c0, %core1] mshape[%c10, %c1] srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.remote_load %alloc[%3, %4] into %1 mcore[%core0, %c0] mshape[%c1, %c10] srcSlice[%c0, %c0] : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #l1> into !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        } {d2m.blocking_loop = 1 : i64}
        d2m.atomic_barrier
        d2m.remote_store %alloc_11[%core0, %core1] from %fold_store : memref<10x10x10x10x!ttcore.tile<32x32, bf16>, #ttcore.shard<20480x2048, 1>, #dram> from !d2m.cb<memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        %rep_drain0 = d2m.wait %fold_cb1 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        d2m.pop %fold_cb1 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        %rep_drain1 = d2m.wait %fold_cb0 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        d2m.pop %fold_cb0 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        %rep_bar = d2m.get_arg(11) : !d2m.local_semaphore
        scf.for %rby = %fold_c0 to %fold_c10 step %fold_c1 {
          scf.for %rbx = %fold_c0 to %fold_c10 step %fold_c1 {
            d2m.semaphore_inc %rep_bar, %fold_c1, core[%rby, %rbx] : !d2m.local_semaphore
          }
        }
        d2m.semaphore_wait %rep_bar, %fold_c100 reset %fold_c0 : !d2m.local_semaphore
      }
      d2m.atomic_barrier
    }, {
    ^datamovement1:
    }, {
      %cst = arith.constant 3.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %cst_17 = arith.constant 2.000000e+00 : bf16
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %0 = d2m.get_cb(5) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %1 = d2m.get_cb(6) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %2 = d2m.get_cb(7) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %3 = d2m.get_cb(8) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %4 = d2m.get_cb(9) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %5 = d2m.get_cb(2) resolution_stage =  compile : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      %6 = d2m.reserve %0 {pin_cb = 3 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
      %7 = d2m.reserve %3 {pin_cb = 6 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %dst = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      %8 = d2m.tile_fill(%cst) : bf16 -> <32x32, bf16>
      memref.store %8, %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      %9 = memref.load %dst[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      memref.store %9, %7[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %3 {pin_cb = 6 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %10 = d2m.wait %3 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %11 = d2m.reserve %4 {pin_cb = 7 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %dst_18 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      %12 = d2m.tile_fill(%cst_17) : bf16 -> <32x32, bf16>
      memref.store %12, %dst_18[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      %13 = memref.load %dst_18[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
      memref.store %13, %11[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      d2m.push %4 {pin_cb = 7 : i64} : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %14 = d2m.wait %4 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %rep_cN = arith.constant 100 : index
      scf.for %rep = %c0 to %rep_cN step %c1 {
        scf.for %arg2 = %c0 to %c10 step %c1 {
          %16 = d2m.wait %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          %17 = d2m.wait %1 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%16, %17 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>, memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) outs(%6 : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, bf16>, %in_19: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
            %18 = "d2m.tile_matmul"(%in, %in_19, %out) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            linalg.yield %18 : !ttcore.tile<32x32, bf16>
          }
          d2m.pop %2 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
          d2m.pop %1 : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
        }
        %15 = d2m.reserve %5 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>> -> memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
        scf.for %arg2 = %c0 to %c10 step %c1 {
          scf.for %arg3 = %c0 to %c10 step %c1 {
            %16 = affine.apply #map6(%arg2)
            %17 = affine.apply #map6(%arg3)
            %18 = memref.load %15[%16, %17] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            %19 = memref.load %10[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %dst_19 = d2m.acquire_dst() : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %20 = "d2m.tile_mul"(%18, %19) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %20, %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %21 = memref.load %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %22 = memref.load %6[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
            %23 = memref.load %14[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
            %24 = "d2m.tile_mul"(%22, %23) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %24, %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %25 = memref.load %dst_19[%c1] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %26 = "d2m.tile_add"(%21, %25) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            memref.store %26, %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            %27 = memref.load %dst_19[%c0] : memref<16x!ttcore.tile<32x32, bf16>, #dst>
            memref.store %27, %15[%arg2, %arg3] : memref<10x10x!ttcore.tile<32x32, bf16>, #l1>
          }
        }
        d2m.push %5 {pin_cb = 2 : i64} : <memref<10x10x!ttcore.tile<32x32, bf16>, #l1>>
      }
      d2m.pop %4 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      d2m.pop %3 : <memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
    }
    %alloc_14 = memref.alloc() {address = 111104 : i64, alignment = 16 : i64} : memref<32x320xbf16, #ttcore.cb_layout<640x2, 1>, #l1>
    %alloc_15 = memref.alloc() {address = 131584 : i64, alignment = 16 : i64} : memref<1x10x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<20480x2048, 1>, #l1>
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
