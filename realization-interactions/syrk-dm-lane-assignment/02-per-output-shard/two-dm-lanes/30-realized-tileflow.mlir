#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ((d1 floordiv 10) * 10 + d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<(d0) -> ((d0 floordiv 10) * 10 + d0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, d3 mod 10)>
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
        %3 = affine.apply #map1(%arg3, %arg2)
        %4 = affine.apply #map2()
        %5 = affine.apply #map2()
        %c0_1 = arith.constant 0 : index
        %c10_2 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        %c0_3 = arith.constant 0 : index
        %6 = affine.apply #map3(%arg2)
        %7 = affine.apply #map2()
        %8 = affine.apply #map2()
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        %c10_6 = arith.constant 10 : index
        %c0_7 = arith.constant 0 : index
        tileplan.temporal_for (%arg6) in (0, 10, 1) {
          %11 = tileflow.read_l1_buffer %2 at(%c0, %c0) epoch(%arg6) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
          %12 = affine.apply #map1(%arg6, %arg3)
          %13 = tileflow.frame_delivery %0 source_core(%3, %12) source_slice(%4, %5) multicast(%c0_1, %arg3, %c10_2, %c1) shape(10, 10) {task.buffer_depth = 2 : i64, task.dm_lane = #task.engine<dm1>} : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %14 = tileflow.frame_at %13 at(%c0_3, %c0_3) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
          %15 = affine.apply #map1(%arg6, %arg3)
          %16 = tileflow.frame_delivery %0 source_core(%6, %15) source_slice(%7, %8) multicast(%arg2, %c0_4, %c1_5, %c10_6) shape(10, 10) {task.buffer_depth = 2 : i64} : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %17 = tileflow.frame_at %16 at(%c0_7, %c0_7) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
          %18 = d2mtile.matmul_block %17, %14 acc %11 block [10, 10, 10] {transpose = true} : !d2mtile.tile<bf16>
          tileflow.write_l1_buffer %2 at(%c0, %c0) epoch(%arg6) to %18 {frameShape = array<i64: 10, 10>} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
        } {epoch_id = 0 : i64}
        %c10_8 = arith.constant 10 : index
        %c10_9 = arith.constant 10 : index
        %9 = d2mtile.fill "3.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
        %10 = d2mtile.fill "2.000000e+00" -> "source_ssa" : !d2mtile.tile<bf16>
        tileplan.shard_for (%arg6) in (%c10_8) axis <row> {
          tileplan.shard_for (%arg7) in (%c10_9) axis <col> {
            %11 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg6, %arg7) tile_symbols() tile_map #map4 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
            %12 = d2mtile.binary tile_mul %11, %9 -> "source_ssa" : !d2mtile.tile<bf16>
            %13 = tileflow.read_l1_buffer %2 at(%arg6, %arg7) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
            %14 = d2mtile.binary tile_mul %13, %10 -> "source_ssa" : !d2mtile.tile<bf16>
            %15 = d2mtile.binary tile_add %12, %14 -> "source_ssa" : !d2mtile.tile<bf16>
            tileplan.write %15, %1[%arg6, %arg7] : !d2mtile.tile<bf16>, !tileplan.shard_view
          }
        }
      }
    }
    return
  }
}

