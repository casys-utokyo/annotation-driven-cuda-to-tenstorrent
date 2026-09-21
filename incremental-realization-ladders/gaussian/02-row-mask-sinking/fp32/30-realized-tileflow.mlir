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
  func.func @_Z10ForwardSubPfS__driver(%arg0: memref<5120x5120xf32> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}, %arg1: memref<5120x5120xf32> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins() outs(%arg0, %arg1 : memref<5120x5120xf32>, memref<5120x5120xf32>) {
      ^bb0(%arg4: memref<5120x5120xf32>, %arg5: memref<5120x5120xf32>):
        %0 = d2mtile.fill "0.000000e+00" : !d2mtile.tile<f32>
        %1 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<5120x5120xf32> -> !tileplan.shard_view
        %2 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [160, 160], partition = [10, 10], physical_grid = [10, 10], shard = [16, 16], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<5120x5120xf32> -> !tileplan.shard_view
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
        %3 = affine.apply #map1(%arg2)
        %4 = affine.apply #map2()
        %c0_9 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10_10 = arith.constant 10 : index
        %c0_11 = arith.constant 0 : index
        %5 = affine.apply #map1(%arg3)
        %6 = affine.apply #map2()
        %c0_12 = arith.constant 0 : index
        %c10_13 = arith.constant 10 : index
        %c1_14 = arith.constant 1 : index
        %c0_15 = arith.constant 0 : index
        %c0_16 = arith.constant 0 : index
        %c0_17 = arith.constant 0 : index
        %c0_18 = arith.constant 0 : index
        tileplan.temporal_for (%arg6) in (0, 5119, 1) {
          %7 = affine.apply #map3(%arg6, %arg2)
          %8 = affine.apply #map3(%arg6, %arg3)
          %9 = affine.apply #map4(%arg6)
          %10 = affine.apply #map4(%arg6)
          %11 = tileflow.frame_delivery %2 source_core(%7, %8) source_slice(%9, %10) multicast(%c0, %c0_4, %c10_5, %c10_6) shape(1, 1) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<f32>>
          %12 = tileflow.frame_at %11 at(%c0_7, %c0_7) : !tileflow.frame<!d2mtile.tile<f32>> -> !d2mtile.tile<f32>
          %13 = affine.apply #map5(%arg6)
          %14 = affine.apply #map5(%arg6)
          %15 = tileflow.broadcast %12 kind scalar from(%13, %14) : !d2mtile.tile<f32> -> !d2mtile.tile<f32>
          %16 = affine.apply #map5(%arg6)
          %17 = d2mtile.unary tile_recip %15 -> "source_ssa" : !d2mtile.tile<f32>
          %18 = affine.apply #map5(%arg6)
          %19 = affine.apply #map3(%arg6, %arg3)
          %20 = affine.apply #map4(%arg6)
          %21 = tileflow.frame_delivery %2 source_core(%3, %19) source_slice(%4, %20) multicast(%arg2, %c0_9, %c1, %c10_10) shape(16, 1) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<f32>>
          %22 = affine.apply #map3(%arg6, %arg2)
          %23 = affine.apply #map4(%arg6)
          %24 = tileflow.frame_delivery %2 source_core(%22, %5) source_slice(%23, %6) multicast(%c0_12, %arg3, %c10_13, %c1_14) shape(1, 16) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<f32>>
          %25 = tileflow.frame_generate (%arg7) in (%c16_3) shape(1, 16) reuse 16 {
            %27 = tileflow.frame_at %24 at(%c0_16, %arg7) : !tileflow.frame<!d2mtile.tile<f32>> -> !d2mtile.tile<f32>
            %28 = tileflow.broadcast %27 kind row from(%18) : !d2mtile.tile<f32> -> !d2mtile.tile<f32>
            tileflow.frame_yield %28 : !d2mtile.tile<f32>
          }
          %26 = tileflow.frame_generate (%arg7) in (%c16_3) shape(1, 16) reuse 16 {
            %27 = affine.apply #map6(%arg6, %arg3, %arg7)
            %28 = tileflow.col_mask symbols(%27) set #set invert : !tileflow.mask<f32>
            tileflow.frame_yield %28 : !tileflow.mask<f32>
          }
          tileplan.shard_for (%arg7) in (%c16_1) axis <row> {
            %27 = tileflow.frame_at %21 at(%arg7, %c0_15) : !tileflow.frame<!d2mtile.tile<f32>> -> !d2mtile.tile<f32>
            %28 = tileflow.broadcast %27 kind col from(%16) : !d2mtile.tile<f32> -> !d2mtile.tile<f32>
            %29 = affine.apply #map7(%arg6, %arg2, %arg7)
            %30 = affine.apply #map7(%arg6, %arg2, %arg7)
            %31 = tileflow.row_mask symbols(%30) set #set : !tileflow.mask<f32>
            %32 = tileflow.compute_select %0 with %28 active(%31 : !tileflow.mask<f32>) : !d2mtile.tile<f32>
            %33 = d2mtile.binary tile_mul %32, %17 -> "source_ssa" : !d2mtile.tile<f32>
            tileplan.shard_for (%arg8) in (%c16_3) axis <col> {
              %34 = tileflow.frame_at %25 at(%c0_17, %arg8) : !tileflow.frame<!d2mtile.tile<f32>> -> !d2mtile.tile<f32>
              %35 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %36 = d2mtile.binary tile_mul %33, %34 -> "source_ssa" : !d2mtile.tile<f32>
              %37 = d2mtile.binary tile_sub %35, %36 -> "source_ssa" : !d2mtile.tile<f32>
              %38 = tileflow.frame_at %26 at(%c0_18, %arg8) : !tileflow.frame<!tileflow.mask<f32>> -> !tileflow.mask<f32>
              %39 = tileflow.compute_select %37 with %35 active(%38 : !tileflow.mask<f32>) {tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
              tileplan.write %39, %2[%arg7, %arg8] : !d2mtile.tile<f32>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<row>}
    }
    return
  }
}

