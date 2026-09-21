#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * 32 - 1)>
#map3 = affine_map<(d0, d1) -> ((d0 * 10 + d1) * -32 + 3198)>
#map4 = affine_map<(d0) -> (d0 mod 10)>
#map5 = affine_map<()[s0] -> (s0)>
#map6 = affine_map<()[s0] -> (s0 - 31)>
#map7 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 - 1) mod 10, d3 mod 10)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, d3 mod 10)>
#map9 = affine_map<(d0, d1, d2, d3) -> ((d2 + d0 * 10 + 1) mod 10, d3 mod 10)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 + 1) mod 10)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d2 mod 10, (d3 + d1 * 10 - 1) mod 10)>
#set = affine_set<(d0)[s0, s1] : (d0 + s0 >= 0, -d0 + s1 >= 0)>
#set1 = affine_set<(d0)[s0] : (d0 + s0 >= 0)>
#set2 = affine_set<(d0)[s0] : (-d0 + s0 >= 0)>
module {
  func.func @_Z15runJacobi2DCUDAiiPfS__driver(%arg0: memref<3200x3200xf32> {tileplan.abi_index = 0 : i64, tileplan.tensor_role = #tileplan.tensor_role<inout>}, %arg1: memref<3200x3200xf32> {tileplan.abi_index = 1 : i64, tileplan.tensor_role = #tileplan.tensor_role<transient>}) attributes {tileplan.selected_compute} {
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    tileplan.device_for (%arg2, %arg3) in (%c10_0, %c10) {
      tileplan.phase name "compute" ins() outs(%arg1, %arg0 : memref<3200x3200xf32>, memref<3200x3200xf32>) {
      ^bb0(%arg4: memref<3200x3200xf32>, %arg5: memref<3200x3200xf32>):
        %0 = tileplan.view %arg4 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<transient>} : memref<3200x3200xf32> -> !tileplan.shard_view
        %1 = tileplan.view %arg5 core_dims(%arg2, %arg3) core_symbols() core_map #map {logical_tiles = [100, 100], partition = [10, 10], physical_grid = [10, 10], shard = [10, 10], tileplan.tensor_role = #tileplan.tensor_role<inout>} : memref<3200x3200xf32> -> !tileplan.shard_view
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
        %5 = d2mtile.fill "2.000000e-01" -> "source_ssa" : !d2mtile.tile<f32>
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
        %c1_23 = arith.constant 1 : index
        %c0_24 = arith.constant 0 : index
        %c10_25 = arith.constant 10 : index
        %14 = affine.apply #map1()
        %c9_26 = arith.constant 9 : index
        %c10_27 = arith.constant 10 : index
        %c0_28 = arith.constant 0 : index
        %c10_29 = arith.constant 10 : index
        %15 = affine.apply #map1()
        %true_30 = arith.constant true
        %true_31 = arith.constant true
        %16 = arith.andi %true_30, %true_31 : i1
        %c0_32 = arith.constant 0 : index
        %c10_33 = arith.constant 10 : index
        %c9_34 = arith.constant 9 : index
        %c10_35 = arith.constant 10 : index
        %17 = affine.apply #map1()
        %true_36 = arith.constant true
        %true_37 = arith.constant true
        %18 = arith.andi %true_36, %true_37 : i1
        %c0_38 = arith.constant 0 : index
        %c10_39 = arith.constant 10 : index
        %c0_40 = arith.constant 0 : index
        %c1_41 = arith.constant 1 : index
        %19 = affine.apply #map1()
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %c0_44 = arith.constant 0 : index
        %c0_45 = arith.constant 0 : index
        %c0_46 = arith.constant 0 : index
        %20 = affine.apply #map2(%arg2, %c0_46)
        %21 = affine.apply #map3(%arg2, %c0_46)
        %22 = tileflow.row_mask symbols(%20, %21) set #set invert : !tileflow.mask<f32>
        %23 = tileflow.row_mask symbols(%20) set #set1 invert : !tileflow.mask<f32>
        %c9_47 = arith.constant 9 : index
        %24 = affine.apply #map2(%arg2, %c9_47)
        %25 = affine.apply #map3(%arg2, %c9_47)
        %26 = tileflow.row_mask symbols(%24, %25) set #set invert : !tileflow.mask<f32>
        %27 = tileflow.row_mask symbols(%25) set #set2 invert : !tileflow.mask<f32>
        %28 = tileflow.frame_pack %23, %27 shape(2, 1) reuse 1000 : !tileflow.mask<f32>, !tileflow.mask<f32> -> !tileflow.frame<!tileflow.mask<f32>>
        %c0_48 = arith.constant 0 : index
        %29 = affine.apply #map2(%arg3, %c0_48)
        %30 = affine.apply #map3(%arg3, %c0_48)
        %31 = tileflow.col_mask symbols(%29, %30) set #set invert : !tileflow.mask<f32>
        %32 = tileflow.col_mask symbols(%29) set #set1 invert : !tileflow.mask<f32>
        %c9_49 = arith.constant 9 : index
        %33 = affine.apply #map2(%arg3, %c9_49)
        %34 = affine.apply #map3(%arg3, %c9_49)
        %35 = tileflow.col_mask symbols(%33, %34) set #set invert : !tileflow.mask<f32>
        %36 = tileflow.col_mask symbols(%34) set #set2 invert : !tileflow.mask<f32>
        %37 = tileflow.frame_pack %32, %36 shape(1, 2) reuse 10 : !tileflow.mask<f32>, !tileflow.mask<f32> -> !tileflow.frame<!tileflow.mask<f32>>
        %c0_50 = arith.constant 0 : index
        %c0_51 = arith.constant 0 : index
        tileplan.temporal_for (%arg6) in (0, 1000, 1) {
          tileplan.shard_for (%arg7) in (%c10_8) axis <row> {
            tileplan.shard_for (%arg8) in (%c10_21) axis <col> {
              %42 = tileflow.local_read %1 tile_dims(%arg7, %arg8) tile_symbols() tile_map #map tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %43 = d2mtile.unary tile_transpose %42 -> "source_ssa" : !d2mtile.tile<f32>
              tileplan.write %43, %2[%arg7, %arg8] : !d2mtile.tile<f32>, !tileplan.l1_buffer
            }
          } {tileplan.publish_persistent}
          %38 = tileflow.frame_delivery %1 source_core(%7, %arg3) source_slice(%c9, %c0_9) shape(1, 10) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<f32>>
          %39 = tileflow.frame_delivery %1 source_core(%9, %arg3) source_slice(%c0_11, %c0_12) shape(1, 10) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<f32>>
          %40 = tileflow.frame_delivery %2 source_core(%arg2, %11) source_slice(%c0_15, %c0_16) shape(10, 1) : !tileplan.l1_buffer -> !tileflow.frame<!d2mtile.tile<f32>>
          %41 = tileflow.frame_delivery %2 source_core(%arg2, %13) source_slice(%c0_19, %c9_20) shape(10, 1) : !tileplan.l1_buffer -> !tileflow.frame<!d2mtile.tile<f32>>
          tileplan.shard_for (%arg7) in (%c10_1) axis <row> {
            %42 = affine.apply #map2(%arg2, %arg7)
            %43 = affine.apply #map3(%arg2, %arg7)
            %44 = arith.cmpi sge, %arg7, %c0_22 : index
            %45 = arith.cmpi slt, %arg7, %c1_23 : index
            %46 = arith.andi %44, %45 : i1
            %47 = arith.cmpi sge, %arg7, %c9_26 : index
            %48 = arith.cmpi slt, %arg7, %c10_27 : index
            %49 = arith.andi %47, %48 : i1
            %50 = arith.cmpi sge, %arg7, %c0_32 : index
            %51 = arith.cmpi slt, %arg7, %c10_33 : index
            %52 = arith.andi %50, %51 : i1
            %53 = affine.apply #map4(%arg7)
            %54 = arith.cmpi sge, %arg7, %c0_38 : index
            %55 = arith.cmpi slt, %arg7, %c10_39 : index
            %56 = arith.andi %54, %55 : i1
            %57 = affine.apply #map4(%arg7)
            %58 = affine.apply #map5()[%42]
            %59 = arith.cmpi slt, %58, %c0_42 : index
            %60 = affine.apply #map6()[%43]
            %61 = arith.cmpi slt, %60, %c0_43 : index
            %62 = arith.ori %59, %61 : i1
            %c0_52 = arith.constant 0 : index
            %c9_53 = arith.constant 9 : index
            %63 = arith.cmpi eq, %arg7, %c9_53 : index
            %c1_54 = arith.constant 1 : index
            %64 = arith.select %63, %c1_54, %c0_52 : index
            %c0_55 = arith.constant 0 : index
            tileplan.shard_for (%arg8) in (%c10_3) axis <col> {
              %65 = arith.cmpi sge, %arg8, %c0_24 : index
              %66 = arith.cmpi slt, %arg8, %c10_25 : index
              %67 = arith.andi %65, %66 : i1
              %68 = arith.andi %46, %67 : i1
              %69 = affine.apply #map4(%arg8)
              %70 = scf.if %68 -> (!d2mtile.tile<f32>) {
                %119 = tileflow.frame_at %38 at(%14, %69) : !tileflow.frame<!d2mtile.tile<f32>> -> !d2mtile.tile<f32>
                scf.yield %119 : !d2mtile.tile<f32>
              } else {
                %119 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map7 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
                scf.yield %119 : !d2mtile.tile<f32>
              }
              %71 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %72 = tileflow.roll %71 axis row by -1 {tileflow.execution = "dm"} : !d2mtile.tile<f32>
              %73 = tileflow.lane_copy %72 with %70 if(%3) axis row src_lane 31 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
              %74 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %75 = arith.cmpi sge, %arg8, %c0_28 : index
              %76 = arith.cmpi slt, %arg8, %c10_29 : index
              %77 = arith.andi %75, %76 : i1
              %78 = arith.andi %49, %77 : i1
              %79 = affine.apply #map4(%arg8)
              %80 = scf.if %78 -> (!d2mtile.tile<f32>) {
                %119 = tileflow.frame_at %39 at(%15, %79) : !tileflow.frame<!d2mtile.tile<f32>> -> !d2mtile.tile<f32>
                scf.yield %119 : !d2mtile.tile<f32>
              } else {
                %119 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map9 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
                scf.yield %119 : !d2mtile.tile<f32>
              }
              %81 = tileflow.roll %74 axis row by 1 {tileflow.execution = "dm"} : !d2mtile.tile<f32>
              %82 = tileflow.lane_copy %81 with %80 if(%4) axis row src_lane 0 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
              %83 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<f32>
              %84 = tileflow.roll %83 axis row by 1 {tileflow.execution = "dm"} : !d2mtile.tile<f32>
              %85 = arith.cmpi sge, %arg8, %c9_34 : index
              %86 = arith.cmpi slt, %arg8, %c10_35 : index
              %87 = arith.andi %85, %86 : i1
              %88 = arith.andi %52, %87 : i1
              %89 = scf.if %88 -> (!d2mtile.tile<f32>) {
                %119 = tileflow.frame_at %40 at(%53, %17) : !tileflow.frame<!d2mtile.tile<f32>> -> !d2mtile.tile<f32>
                scf.yield %119 : !d2mtile.tile<f32>
              } else {
                %119 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map10 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                scf.yield %119 : !d2mtile.tile<f32>
              }
              %90 = tileflow.lane_copy %84 with %89 if(%16) axis row src_lane 0 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
              %91 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<f32>
              %92 = tileflow.roll %91 axis row by -1 {tileflow.execution = "dm"} : !d2mtile.tile<f32>
              %93 = arith.cmpi sge, %arg8, %c0_40 : index
              %94 = arith.cmpi slt, %arg8, %c1_41 : index
              %95 = arith.andi %93, %94 : i1
              %96 = arith.andi %56, %95 : i1
              %97 = scf.if %96 -> (!d2mtile.tile<f32>) {
                %119 = tileflow.frame_at %41 at(%57, %19) : !tileflow.frame<!d2mtile.tile<f32>> -> !d2mtile.tile<f32>
                scf.yield %119 : !d2mtile.tile<f32>
              } else {
                %119 = tileflow.local_read %2 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map11 tile_boundary in_bounds : !tileplan.l1_buffer -> !d2mtile.tile<f32>
                scf.yield %119 : !d2mtile.tile<f32>
              }
              %98 = tileflow.lane_copy %92 with %97 if(%18) axis row src_lane 31 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<f32>
              %99 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              %100 = d2mtile.binary tile_add %98, %90 -> "source_ssa" : !d2mtile.tile<f32>
              %101 = d2mtile.unary tile_transpose %100 -> "source_ssa" : !d2mtile.tile<f32>
              %102 = d2mtile.binary tile_add %99, %82 -> "source_ssa" : !d2mtile.tile<f32>
              %103 = d2mtile.binary tile_add %102, %73 -> "source_ssa" : !d2mtile.tile<f32>
              %104 = d2mtile.binary tile_add %103, %101 -> "source_ssa" : !d2mtile.tile<f32>
              %105 = d2mtile.binary tile_mul %104, %5 -> "source_ssa" : !d2mtile.tile<f32>
              %106 = affine.apply #map2(%arg3, %arg8)
              %107 = affine.apply #map3(%arg3, %arg8)
              %108 = affine.apply #map5()[%106]
              %109 = arith.cmpi slt, %108, %c0_44 : index
              %110 = affine.apply #map6()[%107]
              %111 = arith.cmpi slt, %110, %c0_45 : index
              %112 = arith.ori %109, %111 : i1
              %c0_56 = arith.constant 0 : index
              %c9_57 = arith.constant 9 : index
              %113 = arith.cmpi eq, %arg8, %c9_57 : index
              %c1_58 = arith.constant 1 : index
              %114 = arith.select %113, %c1_58, %c0_56 : index
              %c0_59 = arith.constant 0 : index
              %115 = tileflow.frame_at %28 at(%64, %c0_55) : !tileflow.frame<!tileflow.mask<f32>> -> !tileflow.mask<f32>
              %116 = tileflow.compute_select %105 with %99 active(%115 : !tileflow.mask<f32>) if(%62) {tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
              %117 = tileflow.frame_at %37 at(%c0_59, %114) : !tileflow.frame<!tileflow.mask<f32>> -> !tileflow.mask<f32>
              %118 = tileflow.compute_select %116 with %99 active(%117 : !tileflow.mask<f32>) if(%112) {tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<f32>
              tileplan.write %118, %0[%arg7, %arg8] : !d2mtile.tile<f32>, !tileplan.shard_view
            }
          }
          tileplan.shard_for (%arg7) in (%c10_2) axis <row> {
            tileplan.shard_for (%arg8) in (%c10_4) axis <col> {
              %42 = tileflow.local_read %0 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<f32>
              tileplan.write %42, %1[%arg7, %arg8] : !d2mtile.tile<f32>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

