#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d2 * 32 + d0 * 320, d5 + d3 * 32 + d1 * 320)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4 + d1 * 32 + d0 * 320, d2 * 320 + d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3 + d1 * 32 + d0 * 320, d2 * 320 + d5)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
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
        tileplan.shard_for (%arg6) in (%c10_1) axis <row> {
          %c10_2 = arith.constant 10 : index
          tileplan.shard_for (%arg7) in (%c10_2) axis <col> {
            %2 = tileplan.logical_read %1 access_dims(%arg2, %arg3, %arg6, %arg7) element_map #map1 : !tileplan.shard_view -> !d2mtile.tile<bf16>
            %3 = d2mtile.fill "0.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
            %4 = tileplan.temporal_for (%arg8) in (0, 10, 1) iter_args(%arg9 = %3) -> (!d2mtile.tile<bf16>) {
              %10 = tileplan.logical_read %0 access_dims(%arg3, %arg7, %arg8) element_map #map2 {frameShape = array<i64: 1, 10>} : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %11 = tileplan.logical_read %0 access_dims(%arg2, %arg6, %arg8) element_map #map3 {frameShape = array<i64: 1, 10>} : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %12 = tileplan.compute {indexing_maps = [#map4, #map5, #map6], iterator_types = [#tileplan.iterator_type<parallel>, #tileplan.iterator_type<parallel>, #tileplan.iterator_type<reduction>]} ins(%11, %10 : !d2mtile.tile<bf16>, !d2mtile.tile<bf16>) outs(%arg9 : !d2mtile.tile<bf16>) {
              ^bb0(%arg10: !d2mtile.tile<bf16>, %arg11: !d2mtile.tile<bf16>, %arg12: !d2mtile.tile<bf16>):
                %13 = d2mtile.matmul_block %arg10, %arg11 acc %arg12 block [1, 1, 10] {transpose = true} : !d2mtile.tile<bf16>
                tileplan.yield %13 : !d2mtile.tile<bf16>
              } -> !d2mtile.tile<bf16>
              tileplan.yield %12 : !d2mtile.tile<bf16>
            } {loop_name = "k"}
            %5 = d2mtile.fill "3.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
            %6 = d2mtile.binary tile_mul %2, %5 -> "source_ssa" : !d2mtile.tile<bf16>
            %7 = d2mtile.fill "2.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
            %8 = d2mtile.binary tile_mul %4, %7 -> "source_ssa" : !d2mtile.tile<bf16>
            %9 = d2mtile.binary tile_add %6, %8 -> "source_ssa" : !d2mtile.tile<bf16>
            tileplan.write %9, %1[%arg6, %arg7] : !d2mtile.tile<bf16>, !tileplan.shard_view
          }
        }
      }
    }
    return
  }
}

