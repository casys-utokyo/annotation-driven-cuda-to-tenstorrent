func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xf32> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}, %arg1: memref<5120x5120xf32> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
    tileplan.phase name "compute" ins() outs(%arg0, %arg1 : memref<5120x5120xf32>, memref<5120x5120xf32>) {
    ^bb0(%arg4: memref<5120x5120xf32>, %arg5: memref<5120x5120xf32>):
      %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map affine_map<(d0, d1) -> (d0, d1)> {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<5120x5120xf32> -> !tileplan.shard_view
      %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map affine_map<(d0, d1) -> (d0, d1)> {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<5120x5120xf32> -> !tileplan.shard_view
      tileplan.temporal_for (%arg6) in (0, 5119, 1) {
        %2 = tileplan.logical_read %1 access_dims(%arg6) element_map affine_map<(d0, d1, d2, d3) -> (d0, d0)> : !tileplan.shard_view -> !d2mtile.tile<f32>
        %c16 = arith.constant 16 : index
        tileplan.shard_for (%arg7) in (%c16) axis <row> {
          %3 = tileplan.logical_read %0 access_dims(%arg2, %arg7, %arg6) element_map affine_map<(d0, d1, d2, d3, d4, d5) -> (d3 + d1 * 32 + d0 * 512, d2)> : !tileplan.shard_view -> !d2mtile.tile<f32>
          %4 = tileplan.logical_read %1 access_dims(%arg2, %arg7, %arg6) element_map affine_map<(d0, d1, d2, d3, d4, d5) -> (d3 + d1 * 32 + d0 * 512, d2)> : !tileplan.shard_view -> !d2mtile.tile<f32>
          %c16_2 = arith.constant 16 : index
          tileplan.shard_for (%arg8) in (%c16_2) axis <col> {
            %5 = d2mtile.unary tile_recip %2 -> "source_ssa" : !d2mtile.tile<f32>
            %6 = d2mtile.binary tile_mul %4, %5 -> "source_ssa" : !d2mtile.tile<f32>
            %7 = affine.apply affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>(%arg6, %arg2, %arg7)
            %8 = tileplan.region_predicate symbols(%7) set affine_set<(d0, d1)[s0] : (d0 + s0 >= 0)> : !tileplan.predicate
            tileplan.write_if %6, %0[%arg7, %arg8] when(%8) preserve %3 : !d2mtile.tile<f32> : !d2mtile.tile<f32>, !tileplan.shard_view
          }
        }
        %c16_1 = arith.constant 16 : index
        tileplan.shard_for (%arg7) in (%c16_1) axis <row> {
          %3 = tileplan.logical_read %0 access_dims(%arg2, %arg7, %arg6) element_map affine_map<(d0, d1, d2, d3, d4, d5) -> (d3 + d1 * 32 + d0 * 512, d2)> : !tileplan.shard_view -> !d2mtile.tile<f32>
          %c16_2 = arith.constant 16 : index
          tileplan.shard_for (%arg8) in (%c16_2) axis <col> {
            %4 = tileplan.logical_read %1 access_dims(%arg3, %arg8, %arg6) element_map affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d4 + d1 * 32 + d0 * 512)> : !tileplan.shard_view -> !d2mtile.tile<f32>
            %5 = tileplan.logical_read %1 access_dims(%arg2, %arg3, %arg7, %arg8) element_map affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d2 * 32 + d0 * 512, d5 + d3 * 32 + d1 * 512)> : !tileplan.shard_view -> !d2mtile.tile<f32>
            %6 = d2mtile.binary tile_mul %3, %4 -> "source_ssa" : !d2mtile.tile<f32>
            %7 = d2mtile.binary tile_sub %5, %6 -> "source_ssa" : !d2mtile.tile<f32>
            %8 = affine.apply affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>(%arg6, %arg2, %arg7)
            %9 = tileplan.region_predicate symbols(%8) set affine_set<(d0, d1)[s0] : (d0 + s0 >= 0)> : !tileplan.predicate
            %10 = affine.apply affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32)>(%arg6, %arg3, %arg8)
            %11 = tileplan.region_predicate symbols(%10) set affine_set<(d0, d1)[s0] : (d1 + s0 >= 0)> : !tileplan.predicate
            tileplan.write_if %7, %1[%arg7, %arg8] when(%9, %11) preserve %5 : !d2mtile.tile<f32> : !d2mtile.tile<f32>, !tileplan.shard_view
          }
        }
      } {loop_name = "t"}
    }
  }
  return
}
