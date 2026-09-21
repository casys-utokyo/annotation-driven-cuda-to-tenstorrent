#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0) -> (d0 mod 10)>
#map3 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 - 1) mod 10, d3 mod 10)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, d3 mod 10)>
#map5 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 + 1) mod 10, d3 mod 10)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 + 1) mod 10)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 - 1) mod 10)>
#map8 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * 32 - 1)>
#map9 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * -32 + 3198)>
#map10 = affine_map<()[s0] -> (s0)>
#map11 = affine_map<()[s0] -> (s0 - 31)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xbf16> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}, %arg1: memref<3200x3200xbf16> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins() outs(%arg1, %arg0 : memref<3200x3200xbf16>, memref<3200x3200xbf16>) {
      ^bb0(%arg4: memref<3200x3200xbf16>, %arg5: memref<3200x3200xbf16>):
        %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<3200x3200xbf16> -> !tileplan.shard_view
        %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<3200x3200xbf16> -> !tileplan.shard_view
        %2 = tileplan.converted_view %1 name "frame.converted" : !tileplan.shard_view -> !tileplan.l1_buffer
        %c10_1 = arith.constant 10 : index
        %c10_2 = arith.constant 10 : index
        %c10_3 = arith.constant 10 : index
        %c10_4 = arith.constant 10 : index
        %true = arith.constant true
        %true_5 = arith.constant true
        %3 = arith.andi %true, %true_5 : i1
        %true_6 = arith.constant true
        %true_7 = arith.constant true
        %4 = arith.andi %true_6, %true_7 : i1
        %5 = d2mtile.fill "2.000000e-01" -> "source_ssa" : !d2mtile.tile<bf16>
        %c10_8 = arith.constant 10 : index
        %c-1 = arith.constant -1 : index
        %6 = arith.addi %arg2, %c-1 : index
        %c0 = arith.constant 0 : index
        %7 = arith.maxsi %6, %c0 : index
        %c9 = arith.constant 9 : index
        %c0_9 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %8 = arith.addi %arg2, %c1 : index
        %c9_10 = arith.constant 9 : index
        %9 = arith.minsi %8, %c9_10 : index
        %c0_11 = arith.constant 0 : index
        %c0_12 = arith.constant 0 : index
        %c1_13 = arith.constant 1 : index
        %10 = arith.addi %arg3, %c1_13 : index
        %c9_14 = arith.constant 9 : index
        %11 = arith.minsi %10, %c9_14 : index
        %c0_15 = arith.constant 0 : index
        %c0_16 = arith.constant 0 : index
        %c-1_17 = arith.constant -1 : index
        %12 = arith.addi %arg3, %c-1_17 : index
        %c0_18 = arith.constant 0 : index
        %13 = arith.maxsi %12, %c0_18 : index
        %c0_19 = arith.constant 0 : index
        %c9_20 = arith.constant 9 : index
        %c10_21 = arith.constant 10 : index
        %c0_22 = arith.constant 0 : index
        %true_23 = arith.constant true
        %c0_24 = arith.constant 0 : index
        %true_25 = arith.constant true
        %c0_26 = arith.constant 0 : index
        %c1_27 = arith.constant 1 : index
        %c0_28 = arith.constant 0 : index
        %c10_29 = arith.constant 10 : index
        %14 = affine.apply #map1()
        %c9_30 = arith.constant 9 : index
        %c10_31 = arith.constant 10 : index
        %c0_32 = arith.constant 0 : index
        %c10_33 = arith.constant 10 : index
        %15 = affine.apply #map1()
        %true_34 = arith.constant true
        %true_35 = arith.constant true
        %16 = arith.andi %true_34, %true_35 : i1
        %c0_36 = arith.constant 0 : index
        %c10_37 = arith.constant 10 : index
        %c9_38 = arith.constant 9 : index
        %c10_39 = arith.constant 10 : index
        %17 = affine.apply #map1()
        %true_40 = arith.constant true
        %true_41 = arith.constant true
        %18 = arith.andi %true_40, %true_41 : i1
        %c0_42 = arith.constant 0 : index
        %c10_43 = arith.constant 10 : index
        %c0_44 = arith.constant 0 : index
        %c1_45 = arith.constant 1 : index
        %19 = affine.apply #map1()
        %c0_46 = arith.constant 0 : index
        %true_47 = arith.constant true
        %c0_48 = arith.constant 0 : index
        %true_49 = arith.constant true
        tileplan.temporal_for (%arg6) in (0, 1000, 1) {
          tileplan.shard_for (%arg7) in (%c10_8) axis <row> {
            tileplan.shard_for (%arg8) in (%c10_21) axis <col> {
              %24 = tileflow.local_read %1 tile_dims(%arg7, %arg8) tile_symbols() tile_map #map tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %25 = d2mtile.unary tile_transpose %24 -> "source_ssa" : !d2mtile.tile<bf16>
              tileplan.write %25, %2[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.l1_buffer
            }
          } {tileplan.publish_persistent}
          %20 = tileflow.frame_delivery %1 source_core(%7, %arg3) source_slice(%c9, %c0_9) shape(1, 10) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %21 = tileflow.frame_delivery %1 source_core(%9, %arg3) source_slice(%c0_11, %c0_12) shape(1, 10) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %22 = tileflow.frame_delivery %2 source_core(%arg2, %11) source_slice(%c0_15, %c0_16) shape(10, 1) {task.dm_lane = #task.engine<dm1>} : !tileplan.l1_buffer -> !tileflow.frame<!d2mtile.tile<bf16>>
          %23 = tileflow.frame_delivery %2 source_core(%arg2, %13) source_slice(%c0_19, %c9_20) shape(10, 1) {task.dm_lane = #task.engine<dm1>} : !tileplan.l1_buffer -> !tileflow.frame<!d2mtile.tile<bf16>>
          tileplan.shard_for (%arg7) in (%c10_1) axis <row> {
            %24 = arith.cmpi sge, %arg7, %c0_26 : index
            %25 = arith.cmpi slt, %arg7, %c1_27 : index
            %26 = arith.andi %24, %25 : i1
            %27 = arith.cmpi sge, %arg7, %c9_30 : index
            %28 = arith.cmpi slt, %arg7, %c10_31 : index
            %29 = arith.andi %27, %28 : i1
            %30 = arith.cmpi sge, %arg7, %c0_36 : index
            %31 = arith.cmpi slt, %arg7, %c10_37 : index
            %32 = arith.andi %30, %31 : i1
            %33 = affine.apply #map2(%arg7)
            %34 = arith.cmpi sge, %arg7, %c0_42 : index
            %35 = arith.cmpi slt, %arg7, %c10_43 : index
            %36 = arith.andi %34, %35 : i1
            %37 = affine.apply #map2(%arg7)
            tileplan.shard_for (%arg8) in (%c10_3) axis <col> {
              %38 = arith.cmpi sge, %arg8, %c0_28 : index
              %39 = arith.cmpi slt, %arg8, %c10_29 : index
              %40 = arith.andi %38, %39 : i1
              %41 = arith.andi %26, %40 : i1
              %42 = affine.apply #map2(%arg8)
              %43 = scf.if %41 -> (!d2mtile.tile<bf16>) {
                %79 = tileflow.frame_at %20 at(%14, %42) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %79 : !d2mtile.tile<bf16>
              } else {
                %79 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map3 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
                scf.yield %79 : !d2mtile.tile<bf16>
              }
              %44 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map4 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %45 = tileflow.roll %44 axis row by -1 {tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %46 = tileflow.lane_copy %45 with %43 if(%3) axis row src_lane 31 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %47 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map4 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %48 = arith.cmpi sge, %arg8, %c0_32 : index
              %49 = arith.cmpi slt, %arg8, %c10_33 : index
              %50 = arith.andi %48, %49 : i1
              %51 = arith.andi %29, %50 : i1
              %52 = affine.apply #map2(%arg8)
              %53 = scf.if %51 -> (!d2mtile.tile<bf16>) {
                %79 = tileflow.frame_at %21 at(%15, %52) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %79 : !d2mtile.tile<bf16>
              } else {
                %79 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map5 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
                scf.yield %79 : !d2mtile.tile<bf16>
              }
              %54 = tileflow.roll %47 axis row by 1 {tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %55 = tileflow.lane_copy %54 with %53 if(%4) axis row src_lane 0 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %56 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map4 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %57 = tileflow.roll %56 axis row by 1 {task.dm_lane = #task.engine<dm1>, tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %58 = arith.cmpi sge, %arg8, %c9_38 : index
              %59 = arith.cmpi slt, %arg8, %c10_39 : index
              %60 = arith.andi %58, %59 : i1
              %61 = arith.andi %32, %60 : i1
              %62 = scf.if %61 -> (!d2mtile.tile<bf16>) {
                %79 = tileflow.frame_at %22 at(%33, %17) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %79 : !d2mtile.tile<bf16>
              } else {
                %79 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                scf.yield %79 : !d2mtile.tile<bf16>
              }
              %63 = tileflow.lane_copy %57 with %62 if(%16) axis row src_lane 0 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %64 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map4 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %65 = tileflow.roll %64 axis row by -1 {task.dm_lane = #task.engine<dm1>, tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %66 = arith.cmpi sge, %arg8, %c0_44 : index
              %67 = arith.cmpi slt, %arg8, %c1_45 : index
              %68 = arith.andi %66, %67 : i1
              %69 = arith.andi %36, %68 : i1
              %70 = scf.if %69 -> (!d2mtile.tile<bf16>) {
                %79 = tileflow.frame_at %23 at(%37, %19) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %79 : !d2mtile.tile<bf16>
              } else {
                %79 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map7 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                scf.yield %79 : !d2mtile.tile<bf16>
              }
              %71 = tileflow.lane_copy %65 with %70 if(%18) axis row src_lane 31 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %72 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map4 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %73 = d2mtile.binary tile_add %71, %63 -> "source_ssa" : !d2mtile.tile<bf16>
              %74 = d2mtile.unary tile_transpose %73 -> "source_ssa" : !d2mtile.tile<bf16>
              %75 = d2mtile.binary tile_add %72, %55 -> "source_ssa" : !d2mtile.tile<bf16>
              %76 = d2mtile.binary tile_add %75, %46 -> "source_ssa" : !d2mtile.tile<bf16>
              %77 = d2mtile.binary tile_add %76, %74 -> "source_ssa" : !d2mtile.tile<bf16>
              %78 = d2mtile.binary tile_mul %77, %5 -> "source_ssa" : !d2mtile.tile<bf16>
              tileplan.write %78, %0[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
          tileplan.shard_for (%arg7) in (%c10_2) axis <row> {
            %24 = affine.apply #map8(%arg2, %arg7)
            %25 = affine.apply #map9(%arg2, %arg7)
            %26 = affine.apply #map10()[%24]
            %27 = arith.cmpi sge, %26, %c0_22 : index
            %28 = arith.xori %27, %true_23 : i1
            %29 = affine.apply #map11()[%25]
            %30 = arith.cmpi sge, %29, %c0_24 : index
            %31 = arith.xori %30, %true_25 : i1
            tileplan.shard_for (%arg8) in (%c10_4) axis <col> {
              %32 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map4 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %33 = tileflow.local_read %0 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map4 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %34 = affine.apply #map8(%arg3, %arg8)
              %35 = affine.apply #map9(%arg3, %arg8)
              %36 = affine.apply #map10()[%34]
              %37 = arith.cmpi sge, %36, %c0_46 : index
              %38 = arith.xori %37, %true_47 : i1
              %39 = affine.apply #map11()[%35]
              %40 = arith.cmpi sge, %39, %c0_48 : index
              %41 = arith.xori %40, %true_49 : i1
              %42 = tileflow.lane_copy %33 with %32 if(%28) axis row src_lane 0 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %43 = tileflow.lane_copy %42 with %32 if(%31) axis row src_lane 31 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %44 = tileflow.lane_copy %43 with %32 if(%38) axis col src_lane 0 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %45 = tileflow.lane_copy %44 with %32 if(%41) axis col src_lane 31 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              tileplan.write %45, %1[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

