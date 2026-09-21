#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> ((d1 floordiv 10) * 10 + (d0 floordiv 32) floordiv 16, (d2 floordiv 10) * 10 + (d0 floordiv 32) floordiv 16)>
#map2 = affine_map<(d0) -> ((d0 floordiv 32) mod 16, (d0 floordiv 32) mod 16)>
#map3 = affine_map<(d0) -> (d0 mod 32)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> ((d3 floordiv 10) * 10 + d1 floordiv 16 + d0, (d4 floordiv 10) * 10 + (d2 floordiv 32) floordiv 16)>
#map5 = affine_map<(d0, d1, d2) -> (d1 mod 16, (d2 floordiv 32) mod 16)>
#map6 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> ((d3 floordiv 10) * 10 + (d2 floordiv 32) floordiv 16, (d4 floordiv 10) * 10 + d1 floordiv 16 + d0)>
#map8 = affine_map<(d0, d1, d2) -> ((d2 floordiv 32) mod 16, d1 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d2 mod 16, d3 mod 16)>
#map10 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32)>
#set = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
module {
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}, %arg1: memref<5120x5120xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins() outs(%arg0, %arg1 : memref<5120x5120xbf16>, memref<5120x5120xbf16>) {
      ^bb0(%arg4: memref<5120x5120xbf16>, %arg5: memref<5120x5120xbf16>):
        %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<5120x5120xbf16> -> !tileplan.shard_view
        %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<5120x5120xbf16> -> !tileplan.shard_view
        %c16 = arith.constant 16 : index
        %c16_1 = arith.constant 16 : index
        %c16_2 = arith.constant 16 : index
        %c16_3 = arith.constant 16 : index
        tileplan.temporal_for (%arg6) in (0, 5119, 1) {
          %2 = tileflow.transfer %1 core_dims(%arg6, %arg2, %arg3) core_symbols() core_map #map1 tile_dims(%arg6) tile_symbols() tile_map #map2 core_boundary in_bounds tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
          %3 = affine.apply #map3(%arg6)
          %4 = affine.apply #map3(%arg6)
          %5 = tileflow.broadcast %2 kind scalar from(%3, %4) : !d2mtile.tile<bf16> -> !d2mtile.tile<bf16>
          %6 = affine.apply #map3(%arg6)
          %7 = d2mtile.unary tile_recip %5 -> "source_ssa" : !d2mtile.tile<bf16>
          %8 = affine.apply #map3(%arg6)
          tileplan.shard_for (%arg7) in (%c16_1) axis <row> {
            %9 = tileflow.transfer %1 core_dims(%arg2, %arg7, %arg6, %arg2, %arg3) core_symbols() core_map #map4 tile_dims(%arg2, %arg7, %arg6) tile_symbols() tile_map #map5 core_boundary in_bounds tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
            %10 = tileflow.broadcast %9 kind col from(%6) : !d2mtile.tile<bf16> -> !d2mtile.tile<bf16>
            %11 = d2mtile.binary tile_mul %10, %7 -> "source_ssa" : !d2mtile.tile<bf16>
            %12 = affine.apply #map6(%arg6, %arg2, %arg7)
            %13 = affine.apply #map6(%arg6, %arg2, %arg7)
            %14 = tileflow.row_mask symbols(%13) set #set invert : !tileflow.mask<bf16>
            tileplan.shard_for (%arg8) in (%c16_3) axis <col> {
              %15 = tileflow.transfer %1 core_dims(%arg3, %arg8, %arg6, %arg2, %arg3) core_symbols() core_map #map7 tile_dims(%arg3, %arg8, %arg6) tile_symbols() tile_map #map8 core_boundary in_bounds tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %16 = tileflow.broadcast %15 kind row from(%8) : !d2mtile.tile<bf16> -> !d2mtile.tile<bf16>
              %17 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map9 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %18 = d2mtile.binary tile_mul %11, %16 -> "source_ssa" : !d2mtile.tile<bf16>
              %19 = d2mtile.binary tile_sub %17, %18 -> "source_ssa" : !d2mtile.tile<bf16>
              %20 = affine.apply #map10(%arg6, %arg3, %arg8)
              %21 = tileflow.col_mask symbols(%20) set #set invert : !tileflow.mask<bf16>
              %22 = tileflow.masked_overlay %19 with %17 active(%14, %21) source_map #map symbols() : !d2mtile.tile<bf16>
              tileplan.write %22, %1[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

