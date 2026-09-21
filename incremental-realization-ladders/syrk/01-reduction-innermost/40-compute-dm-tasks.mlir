#map = affine_map<(d0, d1, d2) -> ((d2 floordiv 10) * 10 + d1 floordiv 10 + d0)>
#map1 = affine_map<(d0, d1) -> ((d1 floordiv 10) * 10 + d0 floordiv 10)>
#map2 = affine_map<(d0) -> (d0 mod 10)>
#map3 = affine_map<(d0, d1) -> ((d0 floordiv 10) * 10 + d1 floordiv 10 + d0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 mod 10)>
module {
  func.func @_Z11syrk_kerneliiPfS__driver(%arg0: memref<3200x3200xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<input>}, %arg1: memref<3200x3200xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      %0 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(10, 10) order(0) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10]} : !tileplan.l1_buffer
      %1 = tileplan.persistent_l1 elem bf16 tile(32, 32) domain(10, 10) order(1) {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10]} : !tileplan.l1_buffer
      tileplan.phase name "compute" ins(%arg0 : memref<3200x3200xbf16>) outs(%arg1 : memref<3200x3200xbf16>) {
      ^bb0(%arg4: memref<3200x3200xbf16>, %arg5: memref<3200x3200xbf16>):
        %2 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 1) order(0) : !tileplan.l1_buffer
        %3 = tileplan.l1_buffer role<receive> elem bf16 tile(32, 32) domain(1, 1) order(1) : !tileplan.l1_buffer
        tileplan.bind_persistent %0 : !tileplan.l1_buffer reads(%arg4 : memref<3200x3200xbf16>) writes()
        tileplan.bind_persistent %1 : !tileplan.l1_buffer reads(%arg5 : memref<3200x3200xbf16>) writes(%arg5 : memref<3200x3200xbf16>) {producer = "compute"}
        task.group @dm0_0 engine(<dm0>) domain(<phase>) reads(%0 : !tileplan.l1_buffer [full]) writes(%2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) {
          %c10_1 = arith.constant 10 : index
          %c10_2 = arith.constant 10 : index
          %c0 = arith.constant 0 : index
          %c10_3 = arith.constant 10 : index
          %c1 = arith.constant 1 : index
          %c0_4 = arith.constant 0 : index
          %c1_5 = arith.constant 1 : index
          %c10_6 = arith.constant 10 : index
          tileplan.shard_for (%arg6) in (%c10_1) axis <row> {
            tileplan.shard_for (%arg7) in (%c10_2) axis <col> {
              tileplan.temporal_for (%arg8) in (0, 100, 1) {
                %4 = affine.apply #map(%arg3, %arg7, %arg2)
                %5 = affine.apply #map1(%arg8, %arg3)
                %6 = affine.apply #map2(%arg7)
                %7 = affine.apply #map2(%arg8)
                tileflow.bulk_transfer %0 into %2 source_core(%4, %5) source_slice(%6, %7) multicast(%c0, %arg3, %c10_3, %c1) epoch(%arg8) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
                %8 = affine.apply #map3(%arg2, %arg6)
                %9 = affine.apply #map1(%arg8, %arg3)
                %10 = affine.apply #map2(%arg6)
                %11 = affine.apply #map2(%arg8)
                tileflow.bulk_transfer %0 into %3 source_core(%8, %9) source_slice(%10, %11) multicast(%arg2, %c0_4, %c1_5, %c10_6) epoch(%arg8) {task.engine_label = "dm"} : !tileplan.l1_buffer into !tileplan.l1_buffer
              } {epoch_id = 0 : i64}
            }
          }
        }
        task.group @compute_1 engine(<compute>) domain(<phase>) reads(%1 : !tileplan.l1_buffer [full], %2 : !tileplan.l1_buffer [full], %3 : !tileplan.l1_buffer [full]) writes(%1 : !tileplan.l1_buffer [full]) {
          %c10_1 = arith.constant 10 : index
          %c10_2 = arith.constant 10 : index
          %c0 = arith.constant 0 : index
          %c0_3 = arith.constant 0 : index
          %4 = d2mtile.fill "0.000000e+00" -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
          %5 = d2mtile.fill "3.000000e+00" -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
          %6 = d2mtile.fill "2.000000e+00" -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
          tileplan.shard_for (%arg6) in (%c10_1) axis <row> {
            tileplan.shard_for (%arg7) in (%c10_2) axis <col> {
              %7 = affine.apply #map4(%arg2, %arg3, %arg6, %arg7)
              %8 = affine.apply #map5(%arg2, %arg3, %arg6, %arg7)
              %9 = tileflow.read_l1_buffer %1 at(%7, %8) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %10 = tileplan.temporal_for (%arg8) in (0, 100, 1) iter_args(%arg9 = %4) -> (!d2mtile.tile<bf16>) {
                %14 = tileflow.read_l1_buffer %2 at(%c0, %c0) epoch(%arg8) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %15 = tileflow.read_l1_buffer %3 at(%c0_3, %c0_3) epoch(%arg8) : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                %16 = d2mtile.matmul_block %15, %14 acc %arg9 block [1, 1, 1] {task.engine_label = "compute", transpose = true} : !d2mtile.tile<bf16>
                tileplan.yield %16 : !d2mtile.tile<bf16>
              } {epoch_id = 0 : i64}
              %11 = d2mtile.binary tile_mul %9, %5 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
              %12 = d2mtile.binary tile_mul %10, %6 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
              %13 = d2mtile.binary tile_add %11, %12 -> "source_ssa" {task.engine_label = "compute"} : !d2mtile.tile<bf16>
              tileflow.write_l1_buffer %1 at(%arg6, %arg7) to %13 {task.engine_label = "compute"} : !tileplan.l1_buffer to !d2mtile.tile<bf16>
            }
          }
        }
      }
    }
    return
  }
}

