#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * 32 - 1)>
#map2 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * -32 + 3198)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 floordiv 10) * 10 + (d2 + d0 * 10 - 1) floordiv 10, (d5 floordiv 10) * 10 + d3 floordiv 10 + d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 - 1) mod 10, d3 mod 10)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, d3 mod 10)>
#map6 = affine_map<(d0, d1) -> (d0 + 31, d1)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 floordiv 10) * 10 + (d2 + d0 * 10 + 1) floordiv 10, (d5 floordiv 10) * 10 + d3 floordiv 10 + d1)>
#map8 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 + 1) mod 10, d3 mod 10)>
#map9 = affine_map<(d0, d1) -> (d0 - 31, d1)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 floordiv 10) * 10 + d2 floordiv 10 + d0, (d5 floordiv 10) * 10 + (d3 + d1 * 10 + 1) floordiv 10)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 + 1) mod 10)>
#map12 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 floordiv 10) * 10 + d2 floordiv 10 + d0, (d5 floordiv 10) * 10 + (d3 + d1 * 10 - 1) floordiv 10)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 - 1) mod 10)>
#map14 = affine_map<()[s0] -> (s0)>
#map15 = affine_map<()[s0] -> (s0 - 31)>
#set = affine_set<(d0) : (d0 >= 0, -d0 >= 0)>
#set1 = affine_set<(d0) : (d0 - 31 >= 0, -d0 + 31 >= 0)>
#set2 = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
#set3 = affine_set<(d0)[s0] : (-d0 + s0 >= 0)>
#set4 = affine_set<(d0)[s0, s1] : (d0 + s0 >= 0, -d0 + s1 >= 0)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xf32> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}, %arg1: memref<3200x3200xf32> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins() outs(%arg1, %arg0 : memref<3200x3200xf32>, memref<3200x3200xf32>) {
      ^bb0(%arg4: memref<3200x3200xf32>, %arg5: memref<3200x3200xf32>):
        %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<3200x3200xf32> -> !tileplan.shard_view
        %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<3200x3200xf32> -> !tileplan.shard_view
        %c10_1 = arith.constant 10 : index
        %c10_2 = arith.constant 10 : index
        %c10_3 = arith.constant 10 : index
        %c10_4 = arith.constant 10 : index
        %2 = tileflow.row_mask symbols() set #set : !tileflow.mask<f32>
        %3 = tileflow.row_mask symbols() set #set1 : !tileflow.mask<f32>
        %4 = tileflow.col_mask symbols() set #set1 : !tileflow.mask<f32>
        %5 = tileflow.col_mask symbols() set #set : !tileflow.mask<f32>
        %6 = d2mtile.fill "2.000000e-01" -> "source_ssa" : !d2mtile.tile<f32>
        tileplan.temporal_for (%arg6) in (0, 1000, 1) {
          tileplan.shard_for (%arg7) in (%c10_1) axis <row> {
            %7 = affine.apply #map1(%arg2, %arg7)
            %8 = affine.apply #map2(%arg2, %arg7)
            %9 = tileflow.row_mask symbols(%7) set #set2 invert : !tileflow.mask<f32>
            %10 = tileflow.row_mask symbols(%8) set #set3 invert : !tileflow.mask<f32>
            tileplan.shard_for (%arg8) in (%c10_3) axis <col> {
              %11 = tileflow.transfer %1 core_dims(%arg2, %arg3, %arg7, %arg8, %arg2, %arg3) core_symbols() core_map #map3 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map4 core_boundary clamp tile_boundary clamp : !tileplan.shard_view -> !d2mtile.tile<f32>
              %12 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map5 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %13 = tileflow.roll %12 axis row by -1 : !d2mtile.tile<f32>
              %14 = tileflow.row_copy %13 with %11 active(%2) source_map #map6 symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<f32>
              %15 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map5 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %16 = tileflow.transfer %1 core_dims(%arg2, %arg3, %arg7, %arg8, %arg2, %arg3) core_symbols() core_map #map7 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 core_boundary clamp tile_boundary clamp : !tileplan.shard_view -> !d2mtile.tile<f32>
              %17 = tileflow.roll %15 axis row by 1 : !d2mtile.tile<f32>
              %18 = tileflow.row_copy %17 with %16 active(%3) source_map #map9 symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<f32>
              %19 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map5 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %20 = tileflow.transfer %1 core_dims(%arg2, %arg3, %arg7, %arg8, %arg2, %arg3) core_symbols() core_map #map10 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map11 core_boundary clamp tile_boundary clamp : !tileplan.shard_view -> !d2mtile.tile<f32>
              %21 = tileflow.transpose %19 : !d2mtile.tile<f32>
              %22 = tileflow.roll %21 axis row by 1 : !d2mtile.tile<f32>
              %23 = tileflow.transpose %22 : !d2mtile.tile<f32>
              %24 = tileflow.row_mask symbols() set #set1 : !tileflow.mask<f32>
              %25 = tileflow.transpose %20 {tileflow.conversion_site = #tileflow.conversion_site<sender>} : !d2mtile.tile<f32>
              %26 = tileflow.row_copy %22 with %25 active(%24) source_map #map9 symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<f32>
              %27 = tileflow.transpose %26 : !d2mtile.tile<f32>
              %28 = tileflow.transfer %1 core_dims(%arg2, %arg3, %arg7, %arg8, %arg2, %arg3) core_symbols() core_map #map12 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map13 core_boundary clamp tile_boundary clamp : !tileplan.shard_view -> !d2mtile.tile<f32>
              %29 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map5 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %30 = tileflow.transpose %29 : !d2mtile.tile<f32>
              %31 = tileflow.roll %30 axis row by -1 : !d2mtile.tile<f32>
              %32 = tileflow.transpose %31 : !d2mtile.tile<f32>
              %33 = tileflow.row_mask symbols() set #set : !tileflow.mask<f32>
              %34 = tileflow.transpose %28 {tileflow.conversion_site = #tileflow.conversion_site<sender>} : !d2mtile.tile<f32>
              %35 = tileflow.row_copy %31 with %34 active(%33) source_map #map6 symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<f32>
              %36 = tileflow.transpose %35 : !d2mtile.tile<f32>
              %37 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map5 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %38 = d2mtile.binary tile_add %37, %36 -> "source_ssa" : !d2mtile.tile<f32>
              %39 = d2mtile.binary tile_add %38, %27 -> "source_ssa" : !d2mtile.tile<f32>
              %40 = d2mtile.binary tile_add %39, %18 -> "source_ssa" : !d2mtile.tile<f32>
              %41 = d2mtile.binary tile_add %40, %14 -> "source_ssa" : !d2mtile.tile<f32>
              %42 = d2mtile.binary tile_mul %41, %6 -> "source_ssa" : !d2mtile.tile<f32>
              %43 = affine.apply #map1(%arg3, %arg8)
              %44 = affine.apply #map2(%arg3, %arg8)
              %45 = tileflow.col_mask symbols(%43) set #set2 invert : !tileflow.mask<f32>
              %46 = tileflow.col_mask symbols(%44) set #set3 invert : !tileflow.mask<f32>
              %47 = affine.apply #map14()[%7]
              %c0 = arith.constant 0 : index
              %48 = arith.cmpi slt, %47, %c0 : index
              %49 = affine.apply #map15()[%8]
              %c0_5 = arith.constant 0 : index
              %50 = arith.cmpi slt, %49, %c0_5 : index
              %51 = arith.ori %48, %50 : i1
              %52 = tileflow.row_mask symbols(%7, %8) set #set4 invert : !tileflow.mask<f32>
              %53 = affine.apply #map14()[%43]
              %c0_6 = arith.constant 0 : index
              %54 = arith.cmpi slt, %53, %c0_6 : index
              %55 = affine.apply #map15()[%44]
              %c0_7 = arith.constant 0 : index
              %56 = arith.cmpi slt, %55, %c0_7 : index
              %57 = arith.ori %54, %56 : i1
              %58 = tileflow.col_mask symbols(%43, %44) set #set4 invert : !tileflow.mask<f32>
              %59 = tileflow.compute_select %42 with %37 active(%52 : !tileflow.mask<f32>) if(%51) {tileflow.overlay = #tileflow.overlay_realization<compute_select>, tileflow.realization_engine = #tileflow.engine<compute>} : !d2mtile.tile<f32>
              %60 = tileflow.compute_select %59 with %37 active(%58 : !tileflow.mask<f32>) if(%57) {tileflow.overlay = #tileflow.overlay_realization<compute_select>, tileflow.realization_engine = #tileflow.engine<compute>} : !d2mtile.tile<f32>
              tileplan.write %60, %0[%arg7, %arg8] : !d2mtile.tile<f32>, !tileplan.shard_view
            }
          }
          tileplan.shard_for (%arg7) in (%c10_2) axis <row> {
            tileplan.shard_for (%arg8) in (%c10_4) axis <col> {
              %7 = tileflow.local_read %0 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map5 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              tileplan.write %7, %1[%arg7, %arg8] : !d2mtile.tile<f32>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

