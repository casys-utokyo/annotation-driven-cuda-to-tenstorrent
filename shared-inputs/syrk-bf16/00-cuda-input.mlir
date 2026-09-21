module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @_Z11syrk_kerneliiPfS_(%arg0: i32, %arg1: i32, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3200 = arith.constant 3200 : index
    %c3200_i32 = arith.constant 3200 : i32
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %block_id_x = gpu.block_id  x
    %0 = arith.index_cast %block_id_x : index to i32
    %block_dim_x = gpu.block_dim  x
    %1 = arith.index_cast %block_dim_x : index to i32
    %2 = arith.muli %0, %1 : i32
    %thread_id_x = gpu.thread_id  x
    %3 = arith.index_cast %thread_id_x : index to i32
    %4 = arith.addi %2, %3 : i32
    %block_id_y = gpu.block_id  y
    %5 = arith.index_cast %block_id_y : index to i32
    %block_dim_y = gpu.block_dim  y
    %6 = arith.index_cast %block_dim_y : index to i32
    %7 = arith.muli %5, %6 : i32
    %thread_id_y = gpu.thread_id  y
    %8 = arith.index_cast %thread_id_y : index to i32
    %9 = arith.addi %7, %8 : i32
    %10 = arith.cmpi slt, %9, %c3200_i32 : i32
    %11 = arith.cmpi slt, %4, %c3200_i32 : i32
    %12 = arith.andi %10, %11 : i1
    scf.if %12 {
      %13 = arith.muli %9, %c3200_i32 : i32
      %14 = arith.muli %4, %c3200_i32 : i32
      %15 = scf.for %arg4 = %c0 to %c3200 step %c1 iter_args(%arg5 = %cst_1) -> (f32) {
        %23 = arith.index_cast %arg4 : index to i32
        %24 = arith.addi %13, %23 : i32
        %25 = arith.index_cast %24 : i32 to index
        %26 = memref.load %arg2[%25] : memref<?xf32>
        %27 = arith.addi %14, %23 : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = memref.load %arg2[%28] : memref<?xf32>
        %30 = arith.mulf %26, %29 : f32
        %31 = arith.addf %arg5, %30 : f32
        scf.yield %31 : f32
      }
      %16 = arith.muli %9, %c3200_i32 : i32
      %17 = arith.addi %16, %4 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg3[%18] : memref<?xf32>
      %20 = arith.mulf %19, %cst_0 : f32
      %21 = arith.mulf %15, %cst : f32
      %22 = arith.addf %20, %21 : f32
      memref.store %22, %arg3[%18] : memref<?xf32>
    }
    return
  }
  func.func @_Z8syrkCudaPfS_ii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %c100, %arg11 = %c100, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c32, %arg14 = %c32, %arg15 = %c1) {
      func.call @_Z11syrk_kerneliiPfS_(%arg2, %arg3, %arg0, %arg1) : (i32, i32, memref<?xf32>, memref<?xf32>) -> ()
      gpu.terminator
    }
    %0 = call @cudaThreadSynchronize() : () -> i32
    return
  }
  func.func private @cudaThreadSynchronize() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}
