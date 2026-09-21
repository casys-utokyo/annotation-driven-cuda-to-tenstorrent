#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> ((d2 floordiv 10) * 10 + d0, (d3 floordiv 10) * 10 + d1)>
#map2 = affine_map<(d0, d1) -> (0, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, d3 mod 10)>
module {
  func.func @_Z11syrk_kerneliiPfS__driver(%arg0: memref<3200x3200xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<input>}, %arg1: memref<3200x3200xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins(%arg0 : memref<3200x3200xbf16>) outs(%arg1 : memref<3200x3200xbf16>) {
      ^bb0(%arg4: memref<3200x3200xbf16>, %arg5: memref<3200x3200xbf16>):
        %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<input>} : memref<3200x3200xbf16> -> !tileplan.shard_view
        %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<3200x3200xbf16> -> !tileplan.shard_view
        %2 = tileplan.l1_buffer role<intermediate> elem bf16 tile(32, 32) domain(10, 10) order(0) {frame_axis = #tileplan.axis<row_col>, tt.contraction_accumulator, tt.expanded_accumulator} : !tileplan.l1_buffer
        %c0 = arith.constant 0 : index
        tileplan.temporal_for (%arg6) in (0, 10, 1) {
          %5 = tileflow.read_l1_buffer %2 at(%c0, %c0) epoch(%arg6) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
          %6 = tileflow.transfer %0 core_dims(%arg3, %arg6, %arg2, %arg3) core_symbols() core_map #map1 tile_dims(%arg3, %arg6) tile_symbols() tile_map #map2 core_boundary in_bounds tile_boundary in_bounds {frameShape = array<i64: 10, 10>} : !tileplan.shard_view -> !d2mtile.tile<bf16>
          %7 = tileflow.transfer %0 core_dims(%arg2, %arg6, %arg2, %arg3) core_symbols() core_map #map1 tile_dims(%arg2, %arg6) tile_symbols() tile_map #map2 core_boundary in_bounds tile_boundary in_bounds {frameShape = array<i64: 10, 10>} : !tileplan.shard_view -> !d2mtile.tile<bf16>
          %8 = tileplan.compute {indexing_maps = [#map3, #map4, #map5], iterator_types = [#tileplan.iterator_type<parallel>, #tileplan.iterator_type<parallel>, #tileplan.iterator_type<reduction>]} ins(%7, %6 : !d2mtile.tile<bf16>, !d2mtile.tile<bf16>) outs(%5 : !d2mtile.tile<bf16>) {
          ^bb0(%arg7: !d2mtile.tile<bf16>, %arg8: !d2mtile.tile<bf16>, %arg9: !d2mtile.tile<bf16>):
            %9 = d2mtile.matmul_block %arg7, %arg8 acc %arg9 block [10, 10, 10] {transpose = true} : !d2mtile.tile<bf16>
            tileplan.yield %9 : !d2mtile.tile<bf16>
          } -> !d2mtile.tile<bf16>
          tileflow.write_l1_buffer %2 at(%c0, %c0) epoch(%arg6) to %8 {frameShape = array<i64: 10, 10>} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
        } {epoch_id = 0 : i64}
        %c10_1 = arith.constant 10 : index
        %c10_2 = arith.constant 10 : index
        %3 = d2mtile.fill "3.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
        %4 = d2mtile.fill "2.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
        tileplan.shard_for (%arg6) in (%c10_1) axis <row> {
          tileplan.shard_for (%arg7) in (%c10_2) axis <col> {
            %5 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg6, %arg7) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
            %6 = d2mtile.binary tile_mul %5, %3 -> "source_ssa" : !d2mtile.tile<bf16>
            %7 = tileflow.read_l1_buffer %2 at(%arg6, %arg7) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
            %8 = d2mtile.binary tile_mul %7, %4 -> "source_ssa" : !d2mtile.tile<bf16>
            %9 = d2mtile.binary tile_add %6, %8 -> "source_ssa" : !d2mtile.tile<bf16>
            tileplan.write %9, %1[%arg6, %arg7] : !d2mtile.tile<bf16>, !tileplan.shard_view
          }
        }
      }
    }
    return
  }
}

