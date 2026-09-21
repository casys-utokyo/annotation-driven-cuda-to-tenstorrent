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
        %true = arith.constant true
        %true_5 = arith.constant true
        %2 = arith.andi %true, %true_5 : i1
        %true_6 = arith.constant true
        %true_7 = arith.constant true
        %3 = arith.andi %true_6, %true_7 : i1
        %4 = d2mtile.fill "2.000000e-01" -> "source_ssa" : !d2mtile.tile<bf16>
        %c-1 = arith.constant -1 : index
        %5 = arith.addi %arg2, %c-1 : index
        %c0 = arith.constant 0 : index
        %6 = arith.maxsi %5, %c0 : index
        %c9 = arith.constant 9 : index
        %c0_8 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %7 = arith.addi %arg2, %c1 : index
        %c9_9 = arith.constant 9 : index
        %8 = arith.minsi %7, %c9_9 : index
        %c0_10 = arith.constant 0 : index
        %c0_11 = arith.constant 0 : index
        %c1_12 = arith.constant 1 : index
        %9 = arith.addi %arg3, %c1_12 : index
        %c9_13 = arith.constant 9 : index
        %10 = arith.minsi %9, %c9_13 : index
        %c0_14 = arith.constant 0 : index
        %c0_15 = arith.constant 0 : index
        %c-1_16 = arith.constant -1 : index
        %11 = arith.addi %arg3, %c-1_16 : index
        %c0_17 = arith.constant 0 : index
        %12 = arith.maxsi %11, %c0_17 : index
        %c0_18 = arith.constant 0 : index
        %c9_19 = arith.constant 9 : index
        %c0_20 = arith.constant 0 : index
        %c1_21 = arith.constant 1 : index
        %c0_22 = arith.constant 0 : index
        %c10_23 = arith.constant 10 : index
        %13 = affine.apply #map1()
        %c9_24 = arith.constant 9 : index
        %c10_25 = arith.constant 10 : index
        %c0_26 = arith.constant 0 : index
        %c10_27 = arith.constant 10 : index
        %14 = affine.apply #map1()
        %c0_28 = arith.constant 0 : index
        %c10_29 = arith.constant 10 : index
        %c9_30 = arith.constant 9 : index
        %c10_31 = arith.constant 10 : index
        %15 = affine.apply #map1()
        %true_32 = arith.constant true
        %true_33 = arith.constant true
        %16 = arith.andi %true_32, %true_33 : i1
        %c0_34 = arith.constant 0 : index
        %c10_35 = arith.constant 10 : index
        %c0_36 = arith.constant 0 : index
        %c1_37 = arith.constant 1 : index
        %17 = affine.apply #map1()
        %true_38 = arith.constant true
        %true_39 = arith.constant true
        %18 = arith.andi %true_38, %true_39 : i1
        %c0_40 = arith.constant 0 : index
        %c0_41 = arith.constant 0 : index
        %c0_42 = arith.constant 0 : index
        %c0_43 = arith.constant 0 : index
        %c0_44 = arith.constant 0 : index
        %19 = affine.apply #map2(%arg2, %c0_44)
        %20 = affine.apply #map3(%arg2, %c0_44)
        %21 = tileflow.row_mask symbols(%19, %20) set #set invert : !tileflow.mask<bf16>
        %22 = tileflow.row_mask symbols(%19) set #set1 invert : !tileflow.mask<bf16>
        %c9_45 = arith.constant 9 : index
        %23 = affine.apply #map2(%arg2, %c9_45)
        %24 = affine.apply #map3(%arg2, %c9_45)
        %25 = tileflow.row_mask symbols(%23, %24) set #set invert : !tileflow.mask<bf16>
        %26 = tileflow.row_mask symbols(%24) set #set2 invert : !tileflow.mask<bf16>
        %27 = tileflow.frame_pack %22, %26 shape(2, 1) reuse 1000 : !tileflow.mask<bf16>, !tileflow.mask<bf16> -> !tileflow.frame<!tileflow.mask<bf16>>
        %c0_46 = arith.constant 0 : index
        %28 = affine.apply #map2(%arg3, %c0_46)
        %29 = affine.apply #map3(%arg3, %c0_46)
        %30 = tileflow.col_mask symbols(%28, %29) set #set invert : !tileflow.mask<bf16>
        %31 = tileflow.col_mask symbols(%28) set #set1 invert : !tileflow.mask<bf16>
        %c9_47 = arith.constant 9 : index
        %32 = affine.apply #map2(%arg3, %c9_47)
        %33 = affine.apply #map3(%arg3, %c9_47)
        %34 = tileflow.col_mask symbols(%32, %33) set #set invert : !tileflow.mask<bf16>
        %35 = tileflow.col_mask symbols(%33) set #set2 invert : !tileflow.mask<bf16>
        %36 = tileflow.frame_pack %31, %35 shape(1, 2) reuse 10 : !tileflow.mask<bf16>, !tileflow.mask<bf16> -> !tileflow.frame<!tileflow.mask<bf16>>
        %c0_48 = arith.constant 0 : index
        %c0_49 = arith.constant 0 : index
        tileplan.temporal_for (%arg6) in (0, 1000, 1) {
          %37 = tileflow.frame_delivery %1 source_core(%6, %arg3) source_slice(%c9, %c0_8) shape(1, 10) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %38 = tileflow.frame_delivery %1 source_core(%8, %arg3) source_slice(%c0_10, %c0_11) shape(1, 10) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %39 = tileflow.frame_delivery %1 source_core(%arg2, %10) source_slice(%c0_14, %c0_15) shape(10, 1) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          %40 = tileflow.frame_delivery %1 source_core(%arg2, %12) source_slice(%c0_18, %c9_19) shape(10, 1) : !tileplan.shard_view -> !tileflow.frame<!d2mtile.tile<bf16>>
          tileplan.shard_for (%arg7) in (%c10_1) axis <row> {
            %41 = affine.apply #map2(%arg2, %arg7)
            %42 = affine.apply #map3(%arg2, %arg7)
            %43 = arith.cmpi sge, %arg7, %c0_20 : index
            %44 = arith.cmpi slt, %arg7, %c1_21 : index
            %45 = arith.andi %43, %44 : i1
            %46 = arith.cmpi sge, %arg7, %c9_24 : index
            %47 = arith.cmpi slt, %arg7, %c10_25 : index
            %48 = arith.andi %46, %47 : i1
            %49 = arith.cmpi sge, %arg7, %c0_28 : index
            %50 = arith.cmpi slt, %arg7, %c10_29 : index
            %51 = arith.andi %49, %50 : i1
            %52 = affine.apply #map4(%arg7)
            %53 = arith.cmpi sge, %arg7, %c0_34 : index
            %54 = arith.cmpi slt, %arg7, %c10_35 : index
            %55 = arith.andi %53, %54 : i1
            %56 = affine.apply #map4(%arg7)
            %57 = affine.apply #map5()[%41]
            %58 = arith.cmpi slt, %57, %c0_40 : index
            %59 = affine.apply #map6()[%42]
            %60 = arith.cmpi slt, %59, %c0_41 : index
            %61 = arith.ori %58, %60 : i1
            %c0_50 = arith.constant 0 : index
            %c9_51 = arith.constant 9 : index
            %62 = arith.cmpi eq, %arg7, %c9_51 : index
            %c1_52 = arith.constant 1 : index
            %63 = arith.select %62, %c1_52, %c0_50 : index
            %c0_53 = arith.constant 0 : index
            tileplan.shard_for (%arg8) in (%c10_3) axis <col> {
              %64 = arith.cmpi sge, %arg8, %c0_22 : index
              %65 = arith.cmpi slt, %arg8, %c10_23 : index
              %66 = arith.andi %64, %65 : i1
              %67 = arith.andi %45, %66 : i1
              %68 = affine.apply #map4(%arg8)
              %69 = scf.if %67 -> (!d2mtile.tile<bf16>) {
                %122 = tileflow.frame_at %37 at(%13, %68) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %122 : !d2mtile.tile<bf16>
              } else {
                %122 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map7 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
                scf.yield %122 : !d2mtile.tile<bf16>
              }
              %70 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %71 = tileflow.roll %70 axis row by -1 {tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %72 = tileflow.lane_copy %71 with %69 if(%2) axis row src_lane 31 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %73 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %74 = arith.cmpi sge, %arg8, %c0_26 : index
              %75 = arith.cmpi slt, %arg8, %c10_27 : index
              %76 = arith.andi %74, %75 : i1
              %77 = arith.andi %48, %76 : i1
              %78 = affine.apply #map4(%arg8)
              %79 = scf.if %77 -> (!d2mtile.tile<bf16>) {
                %122 = tileflow.frame_at %38 at(%14, %78) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %122 : !d2mtile.tile<bf16>
              } else {
                %122 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map9 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
                scf.yield %122 : !d2mtile.tile<bf16>
              }
              %80 = tileflow.roll %73 axis row by 1 {tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %81 = tileflow.lane_copy %80 with %79 if(%3) axis row src_lane 0 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %82 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %83 = arith.cmpi sge, %arg8, %c9_30 : index
              %84 = arith.cmpi slt, %arg8, %c10_31 : index
              %85 = arith.andi %83, %84 : i1
              %86 = arith.andi %51, %85 : i1
              %87 = scf.if %86 -> (!d2mtile.tile<bf16>) {
                %122 = tileflow.frame_at %39 at(%52, %15) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %122 : !d2mtile.tile<bf16>
              } else {
                %122 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map10 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
                scf.yield %122 : !d2mtile.tile<bf16>
              }
              %88 = d2mtile.unary tile_transpose %82 -> "source_ssa" : !d2mtile.tile<bf16>
              %89 = tileflow.roll %88 axis row by 1 {tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %90 = d2mtile.unary tile_transpose %87 -> "source_ssa" : !d2mtile.tile<bf16>
              %91 = tileflow.lane_copy %89 with %90 if(%16) axis row src_lane 0 dst_lane 31 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %92 = arith.cmpi sge, %arg8, %c0_36 : index
              %93 = arith.cmpi slt, %arg8, %c1_37 : index
              %94 = arith.andi %92, %93 : i1
              %95 = arith.andi %55, %94 : i1
              %96 = scf.if %95 -> (!d2mtile.tile<bf16>) {
                %122 = tileflow.frame_at %40 at(%56, %17) : !tileflow.frame<!d2mtile.tile<bf16>> -> !d2mtile.tile<bf16>
                scf.yield %122 : !d2mtile.tile<bf16>
              } else {
                %122 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map11 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
                scf.yield %122 : !d2mtile.tile<bf16>
              }
              %97 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %98 = d2mtile.unary tile_transpose %97 -> "source_ssa" : !d2mtile.tile<bf16>
              %99 = tileflow.roll %98 axis row by -1 {tileflow.execution = "dm"} : !d2mtile.tile<bf16>
              %100 = d2mtile.unary tile_transpose %96 -> "source_ssa" : !d2mtile.tile<bf16>
              %101 = tileflow.lane_copy %99 with %100 if(%18) axis row src_lane 31 dst_lane 0 {tileflow.overlay = #tileflow.overlay_realization<dm_copy>} : !d2mtile.tile<bf16>
              %102 = tileflow.local_read %1 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              %103 = d2mtile.binary tile_add %101, %91 -> "source_ssa" : !d2mtile.tile<bf16>
              %104 = d2mtile.unary tile_transpose %103 -> "source_ssa" : !d2mtile.tile<bf16>
              %105 = d2mtile.binary tile_add %102, %81 -> "source_ssa" : !d2mtile.tile<bf16>
              %106 = d2mtile.binary tile_add %105, %72 -> "source_ssa" : !d2mtile.tile<bf16>
              %107 = d2mtile.binary tile_add %106, %104 -> "source_ssa" : !d2mtile.tile<bf16>
              %108 = d2mtile.binary tile_mul %107, %4 -> "source_ssa" : !d2mtile.tile<bf16>
              %109 = affine.apply #map2(%arg3, %arg8)
              %110 = affine.apply #map3(%arg3, %arg8)
              %111 = affine.apply #map5()[%109]
              %112 = arith.cmpi slt, %111, %c0_42 : index
              %113 = affine.apply #map6()[%110]
              %114 = arith.cmpi slt, %113, %c0_43 : index
              %115 = arith.ori %112, %114 : i1
              %c0_54 = arith.constant 0 : index
              %c9_55 = arith.constant 9 : index
              %116 = arith.cmpi eq, %arg8, %c9_55 : index
              %c1_56 = arith.constant 1 : index
              %117 = arith.select %116, %c1_56, %c0_54 : index
              %c0_57 = arith.constant 0 : index
              %118 = tileflow.frame_at %27 at(%63, %c0_53) : !tileflow.frame<!tileflow.mask<bf16>> -> !tileflow.mask<bf16>
              %119 = tileflow.compute_select %108 with %102 active(%118 : !tileflow.mask<bf16>) if(%61) {tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<bf16>
              %120 = tileflow.frame_at %36 at(%c0_57, %117) : !tileflow.frame<!tileflow.mask<bf16>> -> !tileflow.mask<bf16>
              %121 = tileflow.compute_select %119 with %102 active(%120 : !tileflow.mask<bf16>) if(%115) {tileflow.overlay = #tileflow.overlay_realization<compute_select>} : !d2mtile.tile<bf16>
              tileplan.write %121, %0[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
          tileplan.shard_for (%arg7) in (%c10_2) axis <row> {
            tileplan.shard_for (%arg8) in (%c10_4) axis <col> {
              %41 = tileflow.local_read %0 tile_dims(%arg2, %arg3, %arg7, %arg8) tile_symbols() tile_map #map8 tile_boundary in_bounds : !tileplan.shard_view -> !d2mtile.tile<bf16>
              tileplan.write %41, %1[%arg7, %arg8] : !d2mtile.tile<bf16>, !tileplan.shard_view
            }
          }
        }
      } {task.streaming_granularity = #task.streaming_granularity<col>}
    }
    return
  }
}

