#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * 32 - 1)>
#map3 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * -32 + 3198)>
#map4 = affine_map<(d0) -> (d0 mod 10)>
#map5 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 - 1) mod 10, d3 mod 10)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, d3 mod 10)>
#map7 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 + 1) mod 10, d3 mod 10)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 + 1) mod 10)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 - 1) mod 10)>
#map10 = affine_map<()[s0] -> (s0)>
#map11 = affine_map<()[s0] -> (s0 - 31)>
#set = affine_set<(d0)[s0, s1] : (d0 + s0 >= 0, -d0 + s1 >= 0)>
#set1 = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
#set2 = affine_set<(d0)[s0] : (-d0 + s0 >= 0)>
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
        %c0_47 = arith.constant 0 : index
        %c0_48 = arith.constant 0 : index
        %20 = affine.apply #map2(%arg3, %c0_48)
        %21 = affine.apply #map3(%arg3, %c0_48)
        %22 = tileflow.col_mask symbols(%20, %21) set #set invert : !tileflow.mask<bf16>
        %23 = tileflow.col_mask symbols(%20) set #set1 invert : !tileflow.mask<bf16>
        %c9_49 = arith.constant 9 : index
        %24 = affine.apply #map2(%arg3, %c9_49)
        %25 = affine.apply #map3(%arg3, %c9_49)
        %26 = tileflow.col_mask symbols(%24, %25) set #set invert : !tileflow.mask<bf16>
        %27 = tileflow.col_mask symbols(%25) set #set2 invert : !tileflow.mask<bf16>
        %28 = tileflow.frame_pack %23, %27 shape(1, 2) reuse 10 : !tileflow.mask<bf16>, !tileflow.mask<bf16> -> !tileflow.frame<!tileflow.mask<bf16>>
        %c0_50 = arith.constant 0 : index
        tileplan.temporal_for (%arg6) in (0, 1000, 1) {
          tileplan.shard_for (%arg7) in (%c10_8) axis <row> {
            tileplan.shard_for (%arg8) in (%c10_21) axis <col> {
              %33 = tileflow.local_read %1 tile_dims(%arg7, %arg8) tile_symbols() tile_map #map tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %34 = d2mtile.unary tile_transpose %33 -> "source_ssa" : !d2mtile.tile<bf16>
              tileplan.write %34, %2[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.l1_buffer
            }
          } {tileplan.publish_persistent}
          %29 = tileflow.frame_delivery %1 source_core(%7, %arg3) source_slice(%c9, %c0_9) shape(1, 10) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %30 = tileflow.frame_delivery %1 source_core(%9, %arg3) source_slice(%c0_11, %c0_12) shape(1, 10) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %31 = tileflow.frame_delivery %2 source_core(%arg2, %11) source_slice(%c0_15, %c0_16) shape(10, 1) {task.dm_lane = #task.engine<dm1>} : !tileplan.l1_buffer -> !tileflow.frame<!d2mtile.tile<bf16>>
          %32 = tileflow.frame_delivery %2 source_core(%arg2, %13) source_slice(%c0_19, %c9_20) shape(10, 1) {task.dm_lane = #task.engine<dm1>} : !tileplan.l1_buffer -> !tileflow.frame<!d2mtile.tile<bf16>>
          tileplan.shard_for (%arg7) in (%c10_1) axis <row> {
            %33 = arith.cmpi sge, %arg7, %c0_26 : index
            %34 = arith.cmpi slt, %arg7, %c1_27 : index
            %35 = arith.andi %33, %34 : i1
            %36 = arith.cmpi sge, %arg7, %c9_30 : index
            %37 = arith.cmpi slt, %arg7, %c10_31 : index
            %38 = arith.andi %36, %37 : i1
            %39 = arith.cmpi sge, %arg7, %c0_36 : index
            %40 = arith.cmpi slt, %arg7, %c10_37 : index
            %41 = arith.andi %39, %40 : i1
            %42 = affine.apply #map4(%arg7)
            %43 = arith.cmpi sge, %arg7, %c0_42 : index
            %44 = arith.cmpi slt, %arg7, %c10_43 : index
            %45 = arith.andi %43, %44 : i1
            %46 = affine.apply #map4(%arg7)
            tileplan.shard_for (%arg8) in (%c10_3) axis <col> {
              %47 = arith.cmpi sge, %arg8, %c0_28 : index
              %48 = arith.cmpi slt, %arg8, %c10_29 : index
              %49 = arith.andi %47, %48 : i1
              %50 = arith.andi %35, %49 : i1
              %51 = affine.apply #map4(%arg8)
              %52 = scf.if %50 -> (!d2mtile.tile<bf16>) {
                %99 = tileflow.frame_at %29 at(%14, %51) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %99 : !d2mtile.tile<bf16>
              } else {
                %99 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map5 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
                scf.yield %99 : !d2mtile.tile<bf16>
              }
              %53 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %54 = tileflow.roll %53 axis row by -1 {tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %55 = tileflow.lane_copy %54 with %52 if(%3) axis row src_lane 31 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %56 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %57 = arith.cmpi sge, %arg8, %c0_32 : index
              %58 = arith.cmpi slt, %arg8, %c10_33 : index
              %59 = arith.andi %57, %58 : i1
              %60 = arith.andi %38, %59 : i1
              %61 = affine.apply #map4(%arg8)
              %62 = scf.if %60 -> (!d2mtile.tile<bf16>) {
                %99 = tileflow.frame_at %30 at(%15, %61) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %99 : !d2mtile.tile<bf16>
              } else {
                %99 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map7 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
                scf.yield %99 : !d2mtile.tile<bf16>
              }
              %63 = tileflow.roll %56 axis row by 1 {tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %64 = tileflow.lane_copy %63 with %62 if(%4) axis row src_lane 0 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %65 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %66 = tileflow.roll %65 axis row by 1 {task.dm_lane = #task.engine<dm1>, tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %67 = arith.cmpi sge, %arg8, %c9_38 : index
              %68 = arith.cmpi slt, %arg8, %c10_39 : index
              %69 = arith.andi %67, %68 : i1
              %70 = arith.andi %41, %69 : i1
              %71 = scf.if %70 -> (!d2mtile.tile<bf16>) {
                %99 = tileflow.frame_at %31 at(%42, %17) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %99 : !d2mtile.tile<bf16>
              } else {
                %99 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                scf.yield %99 : !d2mtile.tile<bf16>
              }
              %72 = tileflow.lane_copy %66 with %71 if(%16) axis row src_lane 0 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %73 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
              %74 = tileflow.roll %73 axis row by -1 {task.dm_lane = #task.engine<dm1>, tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %75 = arith.cmpi sge, %arg8, %c0_44 : index
              %76 = arith.cmpi slt, %arg8, %c1_45 : index
              %77 = arith.andi %75, %76 : i1
              %78 = arith.andi %45, %77 : i1
              %79 = scf.if %78 -> (!d2mtile.tile<bf16>) {
                %99 = tileflow.frame_at %32 at(%46, %19) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %99 : !d2mtile.tile<bf16>
              } else {
                %99 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map9 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<bf16>
                scf.yield %99 : !d2mtile.tile<bf16>
              }
              %80 = tileflow.lane_copy %74 with %79 if(%18) axis row src_lane 31 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %81 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %82 = d2mtile.binary tile_add %80, %72 -> "source_ssa" : !d2mtile.tile<bf16>
              %83 = d2mtile.unary tile_transpose %82 -> "source_ssa" : !d2mtile.tile<bf16>
              %84 = d2mtile.binary tile_add %81, %64 -> "source_ssa" : !d2mtile.tile<bf16>
              %85 = d2mtile.binary tile_add %84, %55 -> "source_ssa" : !d2mtile.tile<bf16>
              %86 = d2mtile.binary tile_add %85, %83 -> "source_ssa" : !d2mtile.tile<bf16>
              %87 = d2mtile.binary tile_mul %86, %5 -> "source_ssa" : !d2mtile.tile<bf16>
              %88 = affine.apply #map2(%arg3, %arg8)
              %89 = affine.apply #map3(%arg3, %arg8)
              %90 = affine.apply #map10()[%88]
              %91 = arith.cmpi slt, %90, %c0_46 : index
              %92 = affine.apply #map11()[%89]
              %93 = arith.cmpi slt, %92, %c0_47 : index
              %94 = arith.ori %91, %93 : i1
              %c0_51 = arith.constant 0 : index
              %c9_52 = arith.constant 9 : index
              %95 = arith.cmpi eq, %arg8, %c9_52 : index
              %c1_53 = arith.constant 1 : index
              %96 = arith.select %95, %c1_53, %c0_51 : index
              %c0_54 = arith.constant 0 : index
              %97 = tileflow.frame_at %28 at(%c0_54, %96) : !tileflow.frame<!tileflow.mask<bf16>> -> !tileflow.mask<bf16>
              %98 = tileflow.compute_select %87 with %81 active(%97 : !tileflow.mask<bf16>) if(%94) {tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<bf16>
              tileplan.write %98, %0[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
          tileplan.shard_for (%arg7) in (%c10_2) axis <row> {
            %33 = affine.apply #map2(%arg2, %arg7)
            %34 = affine.apply #map3(%arg2, %arg7)
            %35 = affine.apply #map10()[%33]
            %36 = arith.cmpi sge, %35, %c0_22 : index
            %37 = arith.xori %36, %true_23 : i1
            %38 = affine.apply #map11()[%34]
            %39 = arith.cmpi sge, %38, %c0_24 : index
            %40 = arith.xori %39, %true_25 : i1
            tileplan.shard_for (%arg8) in (%c10_4) axis <col> {
              %41 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %42 = tileflow.local_read %0 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map6 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %43 = tileflow.lane_copy %42 with %41 if(%37) axis row src_lane 0 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %44 = tileflow.lane_copy %43 with %41 if(%40) axis row src_lane 31 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              tileplan.write %44, %1[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

