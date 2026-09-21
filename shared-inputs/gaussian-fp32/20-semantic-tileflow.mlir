func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xf32> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}, %arg1: memref<5120x5120xf32> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
  %c10 = arith.constant 10 : index
  %c10_0 = arith.constant 10 : index
  tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
    tileplan.phase name "compute" ins() outs(%arg0, %arg1 : memref<5120x5120xf32>, memref<5120x5120xf32>) {
    ^bb0(%arg4: memref<5120x5120xf32>, %arg5: memref<5120x5120xf32>):
      %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map affine_map<(d0, d1) -> (d0, d1)> {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<5120x5120xf32> -> !tileplan.shard_view
      %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map affine_map<(d0, d1) -> (d0, d1)> {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<5120x5120xf32> -> !tileplan.shard_view
      %c16 = arith.constant 16 : index
      %c16_1 = arith.constant 16 : index
      %c16_2 = arith.constant 16 : index
      %c16_3 = arith.constant 16 : index
      tileplan.temporal_for (%arg6) in (0, 5119, 1) {
        %2 = tileflow.transfer %1 core_dims(%arg6, %arg2, %arg3) core_symbols() core_map affine_map<(d0, d1, d2) -> ((d1 floordiv 10) * 10 + (d0 floordiv 32) floordiv 16, (d2 floordiv 10) * 10 + (d0 floordiv 32) floordiv 16)> tile_dims(%arg6) tile_symbols() tile_map affine_map<(d0) -> ((d0 floordiv 32) mod 16, (d0 floordiv 32) mod 16)> core_boundary in_bounds tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
        %3 = affine.apply affine_map<(d0) -> (d0 mod 32)>(%arg6)
        %4 = affine.apply affine_map<(d0) -> (d0 mod 32)>(%arg6)
        %5 = tileflow.broadcast %2 kind scalar from(%3, %4) : !d2mtile.tile<f32> -> !d2mtile.tile<f32>
        %6 = affine.apply affine_map<(d0) -> (d0 mod 32)>(%arg6)
        %7 = d2mtile.unary tile_recip %5 -> "source_ssa" : !d2mtile.tile<f32>
        %8 = affine.apply affine_map<(d0) -> (d0 mod 32)>(%arg6)
        tileplan.shard_for (%arg7) in (%c16_1) axis <row> {
          %9 = tileflow.transfer %1 core_dims(%arg2, %arg7, %arg6, %arg2, %arg3) core_symbols() core_map affine_map<(d0, d1, d2, d3, d4) -> ((d3 floordiv 10) * 10 + d1 floordiv 16 + d0, (d4 floordiv 10) * 10 + (d2 floordiv 32) floordiv 16)> tile_dims(%arg2, %arg7, %arg6) tile_symbols() tile_map affine_map<(d0, d1, d2) -> (d1 mod 16, (d2 floordiv 32) mod 16)> core_boundary in_bounds tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
          %10 = tileflow.broadcast %9 kind col from(%6) : !d2mtile.tile<f32> -> !d2mtile.tile<f32>
          %11 = d2mtile.binary tile_mul %10, %7 -> "source_ssa" : !d2mtile.tile<f32>
          %12 = affine.apply affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>(%arg6, %arg2, %arg7)
          %13 = affine.apply affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>(%arg6, %arg2, %arg7)
          %14 = tileflow.row_mask symbols(%13) set affine_set<(d0)[s0] : (d0 + s0 >= 0)> invert : !tileflow.mask<f32>
          tileplan.shard_for (%arg8) in (%c16_3) axis <col> {
            %15 = tileflow.transfer %1 core_dims(%arg3, %arg8, %arg6, %arg2, %arg3) core_symbols() core_map affine_map<(d0, d1, d2, d3, d4) -> ((d3 floordiv 10) * 10 + (d2 floordiv 32) floordiv 16, (d4 floordiv 10) * 10 + d1 floordiv 16 + d0)> tile_dims(%arg3, %arg8, %arg6) tile_symbols() tile_map affine_map<(d0, d1, d2) -> ((d2 floordiv 32) mod 16, d1 mod 16)> core_boundary in_bounds tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
            %16 = tileflow.broadcast %15 kind row from(%8) : !d2mtile.tile<f32> -> !d2mtile.tile<f32>
            %17 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map affine_map<(d0, d1, d2, d3) -> (d2 mod 16, d3 mod 16)> tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
            %18 = d2mtile.binary tile_mul %11, %16 -> "source_ssa" : !d2mtile.tile<f32>
            %19 = d2mtile.binary tile_sub %17, %18 -> "source_ssa" : !d2mtile.tile<f32>
            %20 = affine.apply affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32)>(%arg6, %arg3, %arg8)
            %21 = tileflow.col_mask symbols(%20) set affine_set<(d0)[s0] : (d0 + s0 >= 0)> invert : !tileflow.mask<f32>
            %22 = tileflow.masked_overlay %19 with %17 active(%14, %21) source_map affine_map<(d0, d1) -> (d0, d1)> symbols() : !d2mtile.tile<f32>
            tileplan.write %22, %1[%arg7, %arg8] : !d2mtile.tile<f32>, !tileplan.shard_view
          }
        }
      }
    }
  }
  return
}
