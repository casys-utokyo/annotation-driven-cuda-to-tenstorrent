#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> ((d0 floordiv 10) * 10 + d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<(d0, d1) -> ((d1 floordiv 10) * 10 + (d0 floordiv 32) floordiv 16)>
#map4 = affine_map<(d0) -> ((d0 floordiv 32) mod 16)>
#map5 = affine_map<(d0) -> (d0 mod 32)>
#map6 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32)>
#map7 = affine_map<(d0, d1, d2) -> (-d0 + (d1 * 16 + d2) * 32 - 1)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 mod 16, d3 mod 16)>
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
        %c0 = arith.constant 0 : index
        %c0_4 = arith.constant 0 : index
        %c10_5 = arith.constant 10 : index
        %c10_6 = arith.constant 10 : index
        %c0_7 = arith.constant 0 : index
        %c0_8 = arith.constant 0 : index
        %2 = affine.apply #map1(%arg2)
        %3 = affine.apply #map2()
        %c0_9 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10_10 = arith.constant 10 : index
        %c0_11 = arith.constant 0 : index
        %4 = affine.apply #map1(%arg3)
        %5 = affine.apply #map2()
        %c0_12 = arith.constant 0 : index
        %c10_13 = arith.constant 10 : index
        %c1_14 = arith.constant 1 : index
        %c0_15 = arith.constant 0 : index
        %c0_16 = arith.constant 0 : index
        %c0_17 = arith.constant 0 : index
        %c0_18 = arith.constant 0 : index
        tileplan.temporal_for (%arg6) in (0, 5119, 1) {
          %6 = affine.apply #map3(%arg6, %arg2)
          %7 = affine.apply #map3(%arg6, %arg3)
          %8 = affine.apply #map4(%arg6)
          %9 = affine.apply #map4(%arg6)
          %10 = tileflow.frame_delivery %1 source_core(%6, %7) source_slice(%8, %9) multicast(%c0, %c0_4, %c10_5, %c10_6) shape(1, 1) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %11 = tileflow.frame_at %10 at(%c0_7, %c0_7) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
          %12 = affine.apply #map5(%arg6)
          %13 = affine.apply #map5(%arg6)
          %14 = tileflow.broadcast %11 kind scalar from(%12, %13) : !d2mtile.tile<bf16> -> !d2mtile.tile<bf16>
          %15 = affine.apply #map5(%arg6)
          %16 = d2mtile.unary tile_recip %14 -> "source_ssa" : !d2mtile.tile<bf16>
          %17 = affine.apply #map5(%arg6)
          %18 = affine.apply #map3(%arg6, %arg3)
          %19 = affine.apply #map4(%arg6)
          %20 = tileflow.frame_delivery %1 source_core(%2, %18) source_slice(%3, %19) multicast(%arg2, %c0_9, %c1, %c10_10) shape(16, 1) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %21 = affine.apply #map3(%arg6, %arg2)
          %22 = affine.apply #map4(%arg6)
          %23 = tileflow.frame_delivery %1 source_core(%21, %4) source_slice(%22, %5) multicast(%c0_12, %arg3, %c10_13, %c1_14) shape(1, 16) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %24 = tileflow.frame_generate (%arg7) in (%c16_3) shape(1, 16) reuse 16 {
            %26 = tileflow.frame_at %23 at(%c0_16, %arg7) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
            %27 = tileflow.broadcast %26 kind row from(%17) : !d2mtile.tile<bf16> -> !d2mtile.tile<bf16>
            tileflow.frame_yield %27 : !d2mtile.tile<bf16>
          }
          %25 = tileflow.frame_generate (%arg7) in (%c16_3) shape(1, 16) reuse 16 {
            %26 = affine.apply #map6(%arg6, %arg3, %arg7)
            %27 = tileflow.col_mask symbols(%26) set #set invert : !tileflow.mask<bf16>
            tileflow.frame_yield %27 : !tileflow.mask<bf16>
          }
          tileplan.shard_for (%arg7) in (%c16_1) axis <row> {
            %26 = tileflow.frame_at %20 at(%arg7, %c0_15) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
            %27 = tileflow.broadcast %26 kind col from(%15) : !d2mtile.tile<bf16> -> !d2mtile.tile<bf16>
            %28 = d2mtile.binary tile_mul %27, %16 -> "source_ssa" : !d2mtile.tile<bf16>
            %29 = affine.apply #map7(%arg6, %arg2, %arg7)
            %30 = affine.apply #map7(%arg6, %arg2, %arg7)
            %31 = tileflow.row_mask symbols(%30) set #set invert : !tileflow.mask<bf16>
            tileplan.shard_for (%arg8) in (%c16_3) axis <col> {
              %32 = tileflow.frame_at %24 at(%c0_17, %arg8) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
              %33 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %34 = d2mtile.binary tile_mul %28, %32 -> "source_ssa" : !d2mtile.tile<bf16>
              %35 = d2mtile.binary tile_sub %33, %34 -> "source_ssa" : !d2mtile.tile<bf16>
              %36 = tileflow.compute_select %35 with %33 active(%31 : !tileflow.mask<bf16>) {tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<bf16>
              %37 = tileflow.frame_at %25 at(%c0_18, %arg8) : !tileflow.frame<!tileflow.mask<bf16>> -> !tileflow.mask<bf16>
              %38 = tileflow.compute_select %36 with %33 active(%37 : !tileflow.mask<bf16>) {tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<bf16>
              tileplan.write %38, %1[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

