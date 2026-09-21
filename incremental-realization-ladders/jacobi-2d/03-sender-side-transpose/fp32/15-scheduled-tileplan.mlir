#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d2 * 32 + d0 * 320, d5 + d3 * 32 + d1 * 320)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d2 * 32 + d0 * 320 - 1, d5 + d3 * 32 + d1 * 320)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d2 * 32 + d0 * 320 + 1, d5 + d3 * 32 + d1 * 320)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d2 * 32 + d0 * 320, d5 + d3 * 32 + d1 * 320 + 1)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d2 * 32 + d0 * 320, d5 + d3 * 32 + d1 * 320 - 1)>
#map6 = affine_map<(d0, d1, d2) -> ((d1 * 10 + d2) * 32 - 1)>
#map7 = affine_map<(d0, d1, d2) -> ((d1 * 10 + d2) * -32 + 3198)>
#set = affine_set<(d0, d1)[s0] : (d0 + s0 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (-d0 + s0 >= 0)>
#set2 = affine_set<(d0, d1)[s0] : (d1 + s0 >= 0)>
#set3 = affine_set<(d0, d1)[s0] : (-d1 + s0 >= 0)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xf32> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}, %arg1: memref<3200x3200xf32> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins() outs(%arg1, %arg0 : memref<3200x3200xf32>, memref<3200x3200xf32>) {
      ^bb0(%arg4: memref<3200x3200xf32>, %arg5: memref<3200x3200xf32>):
        %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<3200x3200xf32> -> !tileplan.shard_view
        %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<3200x3200xf32> -> !tileplan.shard_view
        tileplan.temporal_for (%arg6) in (0, 1000, 1) {
          %c10_1 = arith.constant 10 : index
          tileplan.shard_for (%arg7) in (%c10_1) axis <row> {
            %c10_3 = arith.constant 10 : index
            tileplan.shard_for (%arg8) in (%c10_3) axis <col> {
              %2 = tileplan.logical_read %0 access_dims(%arg2, %arg3, %arg7, %arg8) element_map #map1 : !tileplan.shard_view -> !d2mtile.tile<f32>
              %3 = tileplan.logical_read %1 access_dims(%arg2, %arg3, %arg7, %arg8) element_map #map2 : !tileplan.shard_view -> !d2mtile.tile<f32>
              %4 = tileplan.logical_read %1 access_dims(%arg2, %arg3, %arg7, %arg8) element_map #map3 : !tileplan.shard_view -> !d2mtile.tile<f32>
              %5 = tileplan.logical_read %1 access_dims(%arg2, %arg3, %arg7, %arg8) element_map #map4 : !tileplan.shard_view -> !d2mtile.tile<f32>
              %6 = tileplan.logical_read %1 access_dims(%arg2, %arg3, %arg7, %arg8) element_map #map5 : !tileplan.shard_view -> !d2mtile.tile<f32>
              %7 = tileplan.logical_read %1 access_dims(%arg2, %arg3, %arg7, %arg8) element_map #map1 : !tileplan.shard_view -> !d2mtile.tile<f32>
              %8 = d2mtile.binary tile_add %7, %6 -> "source_ssa" : !d2mtile.tile<f32>
              %9 = d2mtile.binary tile_add %8, %5 -> "source_ssa" : !d2mtile.tile<f32>
              %10 = d2mtile.binary tile_add %9, %4 -> "source_ssa" : !d2mtile.tile<f32>
              %11 = d2mtile.binary tile_add %10, %3 -> "source_ssa" : !d2mtile.tile<f32>
              %12 = d2mtile.fill "2.000000e-01" -> "source_ssa" : !d2mtile.tile<f32>
              %13 = d2mtile.binary tile_mul %11, %12 -> "source_ssa" : !d2mtile.tile<f32>
              %14 = affine.apply #map6(%arg6, %arg2, %arg7)
              %15 = tileplan.region_predicate symbols(%14) set #set : !tileplan.predicate
              %16 = affine.apply #map7(%arg6, %arg2, %arg7)
              %17 = tileplan.region_predicate symbols(%16) set #set1 : !tileplan.predicate
              %18 = affine.apply #map6(%arg6, %arg3, %arg8)
              %19 = tileplan.region_predicate symbols(%18) set #set2 : !tileplan.predicate
              %20 = affine.apply #map7(%arg6, %arg3, %arg8)
              %21 = tileplan.region_predicate symbols(%20) set #set3 : !tileplan.predicate
              tileplan.write_if %13, %0[%arg7, %arg8] when(%15, %17, %19, %21) preserve %2 : !d2mtile.tile<f32> : !d2mtile.tile<f32>, !tileplan.shard_view
            }
          }
          %c10_2 = arith.constant 10 : index
          tileplan.shard_for (%arg7) in (%c10_2) axis <row> {
            %c10_3 = arith.constant 10 : index
            tileplan.shard_for (%arg8) in (%c10_3) axis <col> {
              %2 = tileplan.logical_read %1 access_dims(%arg2, %arg3, %arg7, %arg8) element_map #map1 : !tileplan.shard_view -> !d2mtile.tile<f32>
              %3 = tileplan.logical_read %0 access_dims(%arg2, %arg3, %arg7, %arg8) element_map #map1 : !tileplan.shard_view -> !d2mtile.tile<f32>
              %4 = affine.apply #map6(%arg6, %arg2, %arg7)
              %5 = tileplan.region_predicate symbols(%4) set #set : !tileplan.predicate
              %6 = affine.apply #map7(%arg6, %arg2, %arg7)
              %7 = tileplan.region_predicate symbols(%6) set #set1 : !tileplan.predicate
              %8 = affine.apply #map6(%arg6, %arg3, %arg8)
              %9 = tileplan.region_predicate symbols(%8) set #set2 : !tileplan.predicate
              %10 = affine.apply #map7(%arg6, %arg3, %arg8)
              %11 = tileplan.region_predicate symbols(%10) set #set3 : !tileplan.predicate
              tileplan.write_if %3, %1[%arg7, %arg8] when(%5, %7, %9, %11) preserve %2 : !d2mtile.tile<f32> : !d2mtile.tile<f32>, !tileplan.shard_view
            }
          }
        } {loop_name = "t"}
      } {tileplan.schedule = [#tileplan.place_transient_guard<producer_writes_abi = 1, consumer_writes_abi = 0, keep = producer>]}
    }
    return
  }
}

