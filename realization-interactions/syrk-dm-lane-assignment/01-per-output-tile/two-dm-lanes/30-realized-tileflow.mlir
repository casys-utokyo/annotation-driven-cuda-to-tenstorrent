#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ((d0 floordiv 10) * 10 + d1 floordiv 10 + d0)>
#map2 = affine_map<(d0) -> (d0 mod 10)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, d3 mod 10)>
#map4 = affine_map<(d0, d1, d2) -> ((d2 floordiv 10) * 10 + d1 floordiv 10 + d0)>
#map5 = affine_map<(d0, d1) -> ((d1 floordiv 10) * 10 + d0 floordiv 10)>
module {
  func.func @_Z11syrk_kerneliiPfS__driver(%arg0: memref<3200x3200xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<input>}, %arg1: memref<3200x3200xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins(%arg0 : memref<3200x3200xbf16>) outs(%arg1 : memref<3200x3200xbf16>) {
      ^bb0(%arg4: memref<3200x3200xbf16>, %arg5: memref<3200x3200xbf16>):
        %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<input>} : memref<3200x3200xbf16> -> !tileplan.shard_view
        %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<3200x3200xbf16> -> !tileplan.shard_view
        %c10_1 = arith.constant 10 : index
        %c10_2 = arith.constant 10 : index
        %2 = d2mtile.fill "0.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
        %3 = d2mtile.fill "3.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
        %4 = d2mtile.fill "2.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
        %c0 = arith.constant 0 : index
        %c10_3 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        %c0_4 = arith.constant 0 : index
        %c0_5 = arith.constant 0 : index
        %c1_6 = arith.constant 1 : index
        %c10_7 = arith.constant 10 : index
        %c0_8 = arith.constant 0 : index
        tileplan.shard_for (%arg6) in (%c10_1) axis <row> {
          %5 = affine.apply #map1(%arg2, %arg6)
          %6 = affine.apply #map2(%arg6)
          tileplan.shard_for (%arg7) in (%c10_2) axis <col> {
            %7 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg6, %arg7) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
            %8 = affine.apply #map4(%arg3, %arg7, %arg2)
            %9 = affine.apply #map2(%arg7)
            %10 = tileplan.temporal_for (%arg8) in (0, 100, 1) iter_args(%arg9 = %2) -> (!d2mtile.tile<bf16>) {
              %14 = affine.apply #map5(%arg8, %arg3)
              %15 = affine.apply #map2(%arg8)
              %16 = tileflow.frame_delivery %0 source_core(%8, %14) source_slice(%9, %15) multicast(%c0, %arg3, %c10_3, %c1) shape(1, 1) {task.buffer_depth = 2 : i64, task.dm_lane = #task.engine<dm1>} : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
              %17 = tileflow.frame_at %16 at(%c0_4, %c0_4) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
              %18 = affine.apply #map5(%arg8, %arg3)
              %19 = affine.apply #map2(%arg8)
              %20 = tileflow.frame_delivery %0 source_core(%5, %18) source_slice(%6, %19) multicast(%arg2, %c0_5, %c1_6, %c10_7) shape(1, 1) {task.buffer_depth = 2 : i64} : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
              %21 = tileflow.frame_at %20 at(%c0_8, %c0_8) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
              %22 = d2mtile.matmul_block %21, %17 acc %arg9 block [1, 1, 1] {transpose = true} : !d2mtile.tile<bf16>
              tileplan.yield %22 : !d2mtile.tile<bf16>
            }
            %11 = d2mtile.binary tile_mul %7, %3 -> "source_ssa" : !d2mtile.tile<bf16>
            %12 = d2mtile.binary tile_mul %10, %4 -> "source_ssa" : !d2mtile.tile<bf16>
            %13 = d2mtile.binary tile_add %11, %12 -> "source_ssa" : !d2mtile.tile<bf16>
            tileplan.write %13, %1[%arg6, %arg7] : !d2mtile.tile<bf16>, !tileplan.shard_view
          }
        }
      }
    }
    return
  }
}

