module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @_Z4Fan1PfS_ii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %thread_id_x = gpu.thread_id  x
    %0 = arith.index_cast %thread_id_x : index to i32
    %block_id_x = gpu.block_id  x
    %1 = arith.index_cast %block_id_x : index to i32
    %block_dim_x = gpu.block_dim  x
    %2 = arith.index_cast %block_dim_x : index to i32
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %0, %3 : i32
    %5 = arith.addi %arg2, %c-1_i32 : i32
    %6 = arith.subi %5, %arg3 : i32
    %7 = arith.cmpi ult, %4, %6 : i32
    scf.if %7 {
      %8 = arith.addi %4, %arg3 : i32
      %9 = arith.addi %8, %c1_i32 : i32
      %10 = arith.muli %arg2, %9 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %arg3 : i32 to index
      %13 = arith.addi %12, %11 : index
      %14 = memref.load %arg1[%13] : memref<?xf32>
      %15 = arith.muli %arg2, %arg3 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.addi %12, %16 : index
      %18 = memref.load %arg1[%17] : memref<?xf32>
      %19 = arith.divf %14, %18 : f32
      memref.store %19, %arg0[%13] : memref<?xf32>
    }
    return
  }
  func.func private @_Z4Fan2PfS_iii(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %thread_id_x = gpu.thread_id  x
    %0 = arith.index_cast %thread_id_x : index to i32
    %block_id_x = gpu.block_id  x
    %1 = arith.index_cast %block_id_x : index to i32
    %block_dim_x = gpu.block_dim  x
    %2 = arith.index_cast %block_dim_x : index to i32
    %3 = arith.muli %1, %2 : i32
    %4 = arith.addi %0, %3 : i32
    %5 = arith.addi %arg2, %c-1_i32 : i32
    %6 = arith.subi %5, %arg4 : i32
    %7 = arith.cmpi ult, %4, %6 : i32
    %thread_id_y = gpu.thread_id  y
    %8 = arith.index_cast %thread_id_y : index to i32
    %block_id_y = gpu.block_id  y
    %9 = arith.index_cast %block_id_y : index to i32
    %block_dim_y = gpu.block_dim  y
    %10 = arith.index_cast %block_dim_y : index to i32
    %11 = arith.muli %9, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.subi %arg2, %arg4 : i32
    %14 = arith.cmpi ult, %12, %13 : i32
    %15 = arith.andi %7, %14 : i1
    scf.if %15 {
      %16 = arith.addi %4, %c1_i32 : i32
      %17 = arith.addi %16, %arg4 : i32
      %18 = arith.muli %arg2, %17 : i32
      %19 = arith.addi %12, %arg4 : i32
      %20 = arith.addi %18, %19 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.addi %18, %arg4 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = memref.load %arg0[%23] : memref<?xf32>
      %25 = arith.muli %arg2, %arg4 : i32
      %26 = arith.addi %25, %19 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = memref.load %arg1[%27] : memref<?xf32>
      %29 = arith.mulf %24, %28 : f32
      %30 = memref.load %arg1[%21] : memref<?xf32>
      %31 = arith.subf %30, %29 : f32
      memref.store %31, %arg1[%21] : memref<?xf32>
    }
    return
  }
  func.func @_Z10ForwardSubPfS_(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c160 = arith.constant 160 : index
    %c5119 = arith.constant 5119 : index
    %c5120_i32 = arith.constant 5120 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c5119 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c160, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c32, %arg13 = %c1, %arg14 = %c1) {
        func.call @_Z4Fan1PfS_ii(%arg0, %arg1, %c5120_i32, %0) : (memref<?xf32>, memref<?xf32>, i32, i32) -> ()
        gpu.terminator
      }
      %1 = func.call @cudaThreadSynchronize() : () -> i32
      %2 = arith.subi %c5120_i32, %0 : i32
      gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c160, %arg10 = %c160, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c32, %arg13 = %c32, %arg14 = %c1) {
        func.call @_Z4Fan2PfS_iii(%arg0, %arg1, %c5120_i32, %2, %0) : (memref<?xf32>, memref<?xf32>, i32, i32, i32) -> ()
        gpu.terminator
      }
      %3 = func.call @cudaThreadSynchronize() : () -> i32
    }
    return
  }
  func.func private @cudaThreadSynchronize() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}
