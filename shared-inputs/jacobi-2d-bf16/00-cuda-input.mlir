module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @_Z21runJacobiCUDA_kernel1iPfS_(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %c-1_i32 = arith.constant -1 : i32
    %c3200_i32 = arith.constant 3200 : i32
    %c3199_i32 = arith.constant 3199 : i32
    %cst = arith.constant 2.000000e-01 : f32
    %c1_i32 = arith.constant 1 : i32
    %block_id_y = gpu.block_id  y
    %0 = arith.index_cast %block_id_y : index to i32
    %block_dim_y = gpu.block_dim  y
    %1 = arith.index_cast %block_dim_y : index to i32
    %2 = arith.muli %0, %1 : i32
    %thread_id_y = gpu.thread_id  y
    %3 = arith.index_cast %thread_id_y : index to i32
    %4 = arith.addi %2, %3 : i32
    %block_id_x = gpu.block_id  x
    %5 = arith.index_cast %block_id_x : index to i32
    %block_dim_x = gpu.block_dim  x
    %6 = arith.index_cast %block_dim_x : index to i32
    %7 = arith.muli %5, %6 : i32
    %thread_id_x = gpu.thread_id  x
    %8 = arith.index_cast %thread_id_x : index to i32
    %9 = arith.addi %7, %8 : i32
    %10 = arith.cmpi sge, %4, %c1_i32 : i32
    %11 = arith.cmpi slt, %4, %c3199_i32 : i32
    %12 = arith.andi %10, %11 : i1
    %13 = arith.cmpi sge, %9, %c1_i32 : i32
    %14 = arith.cmpi slt, %9, %c3199_i32 : i32
    %15 = arith.andi %13, %14 : i1
    %16 = arith.andi %12, %15 : i1
    scf.if %16 {
      %17 = arith.muli %4, %c3200_i32 : i32
      %18 = arith.addi %17, %9 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = memref.load %arg1[%19] : memref<?xf32>
      %21 = arith.addi %9, %c-1_i32 : i32
      %22 = arith.addi %17, %21 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = memref.load %arg1[%23] : memref<?xf32>
      %25 = arith.addf %20, %24 : f32
      %26 = arith.addi %9, %c1_i32 : i32
      %27 = arith.addi %17, %26 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = memref.load %arg1[%28] : memref<?xf32>
      %30 = arith.addf %25, %29 : f32
      %31 = arith.addi %4, %c1_i32 : i32
      %32 = arith.muli %31, %c3200_i32 : i32
      %33 = arith.addi %32, %9 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = memref.load %arg1[%34] : memref<?xf32>
      %36 = arith.addf %30, %35 : f32
      %37 = arith.addi %4, %c-1_i32 : i32
      %38 = arith.muli %37, %c3200_i32 : i32
      %39 = arith.addi %38, %9 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = memref.load %arg1[%40] : memref<?xf32>
      %42 = arith.addf %36, %41 : f32
      %43 = arith.mulf %42, %cst : f32
      memref.store %43, %arg2[%19] : memref<?xf32>
    }
    return
  }
  func.func private @_Z21runJacobiCUDA_kernel2iPfS_(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %c3200_i32 = arith.constant 3200 : i32
    %c3199_i32 = arith.constant 3199 : i32
    %c1_i32 = arith.constant 1 : i32
    %block_id_y = gpu.block_id  y
    %0 = arith.index_cast %block_id_y : index to i32
    %block_dim_y = gpu.block_dim  y
    %1 = arith.index_cast %block_dim_y : index to i32
    %2 = arith.muli %0, %1 : i32
    %thread_id_y = gpu.thread_id  y
    %3 = arith.index_cast %thread_id_y : index to i32
    %4 = arith.addi %2, %3 : i32
    %block_id_x = gpu.block_id  x
    %5 = arith.index_cast %block_id_x : index to i32
    %block_dim_x = gpu.block_dim  x
    %6 = arith.index_cast %block_dim_x : index to i32
    %7 = arith.muli %5, %6 : i32
    %thread_id_x = gpu.thread_id  x
    %8 = arith.index_cast %thread_id_x : index to i32
    %9 = arith.addi %7, %8 : i32
    %10 = arith.cmpi sge, %4, %c1_i32 : i32
    %11 = arith.cmpi slt, %4, %c3199_i32 : i32
    %12 = arith.andi %10, %11 : i1
    %13 = arith.cmpi sge, %9, %c1_i32 : i32
    %14 = arith.cmpi slt, %9, %c3199_i32 : i32
    %15 = arith.andi %13, %14 : i1
    %16 = arith.andi %12, %15 : i1
    scf.if %16 {
      %17 = arith.muli %4, %c3200_i32 : i32
      %18 = arith.addi %17, %9 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = memref.load %arg2[%19] : memref<?xf32>
      memref.store %20, %arg1[%19] : memref<?xf32>
    }
    return
  }
  func.func @_Z15runJacobi2DCUDAiiPfS_(%arg0: i32, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c100 = arith.constant 100 : index
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c1000 step %c1 {
      gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c100, %arg12 = %c100, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c32, %arg15 = %c32, %arg16 = %c1) {
        func.call @_Z21runJacobiCUDA_kernel1iPfS_(%arg1, %arg2, %arg3) : (i32, memref<?xf32>, memref<?xf32>) -> ()
        gpu.terminator
      }
      %0 = func.call @cudaThreadSynchronize() : () -> i32
      gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c100, %arg12 = %c100, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c32, %arg15 = %c32, %arg16 = %c1) {
        func.call @_Z21runJacobiCUDA_kernel2iPfS_(%arg1, %arg2, %arg3) : (i32, memref<?xf32>, memref<?xf32>) -> ()
        gpu.terminator
      }
      %1 = func.call @cudaThreadSynchronize() : () -> i32
    }
    return
  }
  func.func private @cudaThreadSynchronize() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}
