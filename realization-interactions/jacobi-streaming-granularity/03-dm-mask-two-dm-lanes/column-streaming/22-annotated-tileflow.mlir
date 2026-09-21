#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 floordiv 10) * 10 + (d2 + d0 * 10 - 1) floordiv 10, (d5 floordiv 10) * 10 + d3 floordiv 10 + d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 - 1) mod 10, d3 mod 10)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, d3 mod 10)>
#map4 = affine_map<(d0, d1) -> (d0 + 31, d1)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 floordiv 10) * 10 + (d2 + d0 * 10 + 1) floordiv 10, (d5 floordiv 10) * 10 + d3 floordiv 10 + d1)>
#map6 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 + 1) mod 10, d3 mod 10)>
#map7 = affine_map<(d0, d1) -> (d0 - 31, d1)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 floordiv 10) * 10 + d2 floordiv 10 + d0, (d5 floordiv 10) * 10 + (d3 + d1 * 10 + 1) floordiv 10)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 + 1) mod 10)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 floordiv 10) * 10 + d2 floordiv 10 + d0, (d5 floordiv 10) * 10 + (d3 + d1 * 10 - 1) floordiv 10)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 - 1) mod 10)>
#map12 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * 32 - 1)>
#map13 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * -32 + 3198)>
#set = affine_set<(d0) : (d0 >= 0, -d0 >= 0)>
#set1 = affine_set<(d0) : (d0 - 31 >= 0, -d0 + 31 >= 0)>
#set2 = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
#set3 = affine_set<(d0)[s0] : (-d0 + s0 >= 0)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}, %arg1: memref<3200x3200xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins() outs(%arg1, %arg0 : memref<3200x3200xbf16>, memref<3200x3200xbf16>) {
      ^bb0(%arg4: memref<3200x3200xbf16>, %arg5: memref<3200x3200xbf16>):
        %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<3200x3200xbf16> -> !tileplan.shard_view
        %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<3200x3200xbf16> -> !tileplan.shard_view
        %c10_1 = arith.constant 10 : index
        %c10_2 = arith.constant 10 : index
        %c10_3 = arith.constant 10 : index
        %c10_4 = arith.constant 10 : index
        %2 = tileflow.row_mask symbols() set #set : !tileflow.mask<bf16>
        %3 = tileflow.row_mask symbols() set #set1 : !tileflow.mask<bf16>
        %4 = tileflow.col_mask symbols() set #set1 : !tileflow.mask<bf16>
        %5 = tileflow.col_mask symbols() set #set : !tileflow.mask<bf16>
        %6 = d2mtile.fill "2.000000e-01" -> "source_ssa" : !d2mtile.tile<bf16>
        tileplan.temporal_for (%arg6) in (0, 1000, 1) {
          tileplan.shard_for (%arg7) in (%c10_1) axis <row> {
            tileplan.shard_for (%arg8) in (%c10_3) axis <col> {
              %7 = tileflow.transfer %1 core_dims(%arg2, %arg3, %arg7, %arg8, %arg2, %arg3) core_symbols() core_map #map1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map2 core_boundary clamp tile_boundary clamp : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %8 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %9 = tileflow.roll %8 axis row by -1 : !d2mtile.tile<bf16>
              %10 = tileflow.row_copy %9 with %7 active(%2) source_map #map4 symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<bf16>
              %11 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %12 = tileflow.transfer %1 core_dims(%arg2, %arg3, %arg7, %arg8, %arg2, %arg3) core_symbols() core_map #map5 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 core_boundary clamp tile_boundary clamp : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %13 = tileflow.roll %11 axis row by 1 : !d2mtile.tile<bf16>
              %14 = tileflow.row_copy %13 with %12 active(%3) source_map #map7 symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<bf16>
              %15 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %16 = tileflow.transfer %1 core_dims(%arg2, %arg3, %arg7, %arg8, %arg2, %arg3) core_symbols() core_map #map8 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map9 core_boundary clamp tile_boundary clamp : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %17 = tileflow.transpose %15 : !d2mtile.tile<bf16>
              %18 = tileflow.roll %17 axis row by 1 : !d2mtile.tile<bf16>
              %19 = tileflow.transpose %18 : !d2mtile.tile<bf16>
              %20 = tileflow.row_mask symbols() set #set1 : !tileflow.mask<bf16>
              %21 = tileflow.transpose %16 {tileflow.conversion_site = #tileflow.conversion_site<sender>} : !d2mtile.tile<bf16>
              %22 = tileflow.row_copy %18 with %21 active(%20) source_map #map7 symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<bf16>
              %23 = tileflow.transpose %22 : !d2mtile.tile<bf16>
              %24 = tileflow.transfer %1 core_dims(%arg2, %arg3, %arg7, %arg8, %arg2, %arg3) core_symbols() core_map #map10 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map11 core_boundary clamp tile_boundary clamp : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %25 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %26 = tileflow.transpose %25 : !d2mtile.tile<bf16>
              %27 = tileflow.roll %26 axis row by -1 : !d2mtile.tile<bf16>
              %28 = tileflow.transpose %27 : !d2mtile.tile<bf16>
              %29 = tileflow.row_mask symbols() set #set : !tileflow.mask<bf16>
              %30 = tileflow.transpose %24 {tileflow.conversion_site = #tileflow.conversion_site<sender>} : !d2mtile.tile<bf16>
              %31 = tileflow.row_copy %27 with %30 active(%29) source_map #map4 symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<bf16>
              %32 = tileflow.transpose %31 : !d2mtile.tile<bf16>
              %33 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %34 = d2mtile.binary tile_add %33, %32 -> "source_ssa" : !d2mtile.tile<bf16>
              %35 = d2mtile.binary tile_add %34, %23 -> "source_ssa" : !d2mtile.tile<bf16>
              %36 = d2mtile.binary tile_add %35, %14 -> "source_ssa" : !d2mtile.tile<bf16>
              %37 = d2mtile.binary tile_add %36, %10 -> "source_ssa" : !d2mtile.tile<bf16>
              %38 = d2mtile.binary tile_mul %37, %6 -> "source_ssa" : !d2mtile.tile<bf16>
              tileplan.write %38, %0[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
          tileplan.shard_for (%arg7) in (%c10_2) axis <row> {
            %7 = affine.apply #map12(%arg2, %arg7)
            %8 = affine.apply #map13(%arg2, %arg7)
            %9 = tileflow.row_mask symbols(%7) set #set2 invert : !tileflow.mask<bf16>
            %10 = tileflow.row_mask symbols(%8) set #set3 invert : !tileflow.mask<bf16>
            tileplan.shard_for (%arg8) in (%c10_4) axis <col> {
              %11 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %12 = tileflow.local_read %0 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %13 = affine.apply #map12(%arg3, %arg8)
              %14 = affine.apply #map13(%arg3, %arg8)
              %15 = tileflow.col_mask symbols(%13) set #set2 invert : !tileflow.mask<bf16>
              %16 = tileflow.col_mask symbols(%14) set #set3 invert : !tileflow.mask<bf16>
              %17 = tileflow.row_copy %12 with %11 active(%9) source_map #map symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<bf16>
              %18 = tileflow.row_copy %17 with %11 active(%10) source_map #map symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<bf16>
              %19 = tileflow.col_copy %18 with %11 active(%15) source_map #map symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<bf16>
              %20 = tileflow.col_copy %19 with %11 active(%16) source_map #map symbols() {tileflow.realization_engine = #tileflow.engine<dm>} : !d2mtile.tile<bf16>
              tileplan.write %20, %1[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

