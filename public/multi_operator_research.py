"""
多算子CUDA调研工具
调研算子：addmm, softmax, layernorm, transpose
包括：实现原理、CUDA kernel、CUDA Runtime API使用情况
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import json
import os
from collections import defaultdict
import numpy as np


class MultiOperatorResearch:
    """多算子调研工具包"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，部分功能将无法使用")
            return
        
        # 创建测试数据
        self.batch_size = 32
        self.seq_len = 128
        self.hidden_size = 768
        
        print(f"\n测试配置:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Sequence length: {self.seq_len}")
        print(f"  Hidden size: {self.hidden_size}")
        
    def warm_up(self, func, iterations=10):
        """GPU预热"""
        for _ in range(iterations):
            func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def research_addmm(self):
        """
        调研 addmm 算子
        addmm: 矩阵乘加操作 (Add Matrix Multiply)
        公式: out = beta * input + alpha * (mat1 @ mat2)
        """
        print("\n" + "="*80)
        print("1. ADDMM 算子调研")
        print("="*80)
        
        print("\n【数学原理】")
        print("addmm 执行矩阵乘加操作:")
        print("  out = β * input + α * (mat1 @ mat2)")
        print("其中:")
        print("  - mat1: [m, k] 矩阵")
        print("  - mat2: [k, n] 矩阵")
        print("  - input: [m, n] 矩阵（可广播）")
        print("  - α, β: 标量系数")
        print("\n典型应用场景:")
        print("  - 全连接层: y = Wx + b")
        print("  - 线性变换")
        print("  - Transformer的线性投影")
        
        print("\n【PyTorch实现路径】")
        print("Python层:")
        print("  torch.addmm(bias, input, weight)")
        print("    ↓")
        print("C++层 (torch/csrc/):")
        print("  at::addmm")
        print("    ↓")
        print("ATen层 (aten/src/ATen/):")
        print("  aten/src/ATen/native/LinearAlgebra.cpp")
        print("  aten/src/ATen/native/cuda/Blas.cpp")
        print("    ↓")
        print("CUDA实现:")
        print("  调用cuBLAS库: cublasSgemm / cublasGemmEx")
        
        print("\n【源码位置】")
        print("1. 算子声明:")
        print("   aten/src/ATen/native/native_functions.yaml")
        print("   - func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)")
        print("\n2. CPU实现:")
        print("   aten/src/ATen/native/LinearAlgebra.cpp")
        print("   - addmm_impl_cpu")
        print("\n3. CUDA实现:")
        print("   aten/src/ATen/native/cuda/Blas.cpp")
        print("   - addmm_out_cuda_impl")
        print("   调用: aten/src/ATen/cuda/CUDABlas.cpp")
        print("   - gemm<scalar_t>() 函数")
        print("\n4. cuBLAS调用:")
        print("   aten/src/ATen/cuda/CUDABlas.cpp")
        print("   - cublasSgemm (单精度)")
        print("   - cublasDgemm (双精度)")
        print("   - cublasGemmEx (混合精度)")
        
        print("\n【CUDA并行化策略】")
        print("1. 使用cuBLAS库进行矩阵乘法")
        print("   - 高度优化的BLAS库")
        print("   - 自动选择最优算法")
        print("   - 支持Tensor Core加速")
        print("\n2. 矩阵分块 (Tiling)")
        print("   - 将大矩阵分成小块")
        print("   - 每个block处理一个tile")
        print("   - 使用shared memory缓存")
        print("\n3. 并行维度")
        print("   - Grid: (M/tile_m, N/tile_n)")
        print("   - Block: (tile_size, tile_size)")
        print("   - 每个线程计算多个元素")
        
        print("\n【性能测试】")
        # 创建测试数据
        m, k, n = 1024, 768, 512
        input_bias = torch.randn(m, n).cuda()
        mat1 = torch.randn(m, k).cuda()
        mat2 = torch.randn(k, n).cuda()
        
        # 预热
        self.warm_up(lambda: torch.addmm(input_bias, mat1, mat2))
        
        # 性能测试
        iterations = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            result = torch.addmm(input_bias, mat1, mat2)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_time = start.elapsed_time(end) / iterations
        
        # 计算理论性能
        flops = 2 * m * k * n  # 矩阵乘法的浮点运算数
        tflops = (flops * iterations) / (start.elapsed_time(end) / 1000) / 1e12
        
        print(f"\n矩阵尺寸: [{m}, {k}] @ [{k}, {n}]")
        print(f"平均时间: {elapsed_time:.4f} ms")
        print(f"吞吐量: {tflops:.2f} TFLOPS")
        
        return {
            'operation': 'addmm',
            'avg_time_ms': elapsed_time,
            'throughput_tflops': tflops
        }
    
    def research_softmax(self):
        """
        调研 softmax 算子
        softmax: 归一化指数函数
        """
        print("\n" + "="*80)
        print("2. SOFTMAX 算子调研")
        print("="*80)
        
        print("\n【数学原理】")
        print("Softmax 将输入转换为概率分布:")
        print("  softmax(x_i) = exp(x_i) / Σ exp(x_j)")
        print("\n数值稳定版本:")
        print("  softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))")
        print("\n计算步骤:")
        print("  1. 找到最大值: m = max(x)")
        print("  2. 计算指数: e_i = exp(x_i - m)")
        print("  3. 求和: s = Σ e_i")
        print("  4. 归一化: y_i = e_i / s")
        print("\n应用场景:")
        print("  - Attention机制中的权重归一化")
        print("  - 分类任务的输出层")
        print("  - 温度缩放的概率分布")
        
        print("\n【PyTorch实现路径】")
        print("Python层:")
        print("  torch.nn.functional.softmax(x, dim=-1)")
        print("    ↓")
        print("C++层:")
        print("  at::softmax")
        print("    ↓")
        print("ATen层:")
        print("  aten/src/ATen/native/SoftMax.cpp")
        print("  aten/src/ATen/native/cuda/SoftMax.cu")
        print("    ↓")
        print("CUDA Kernel:")
        print("  cunn_SoftMaxForward")
        print("  fast_softmax_kernel (优化版本)")
        
        print("\n【源码位置】")
        print("1. 算子声明:")
        print("   aten/src/ATen/native/native_functions.yaml")
        print("   - func: softmax(Tensor self, int dim, ScalarType? dtype=None)")
        print("\n2. CPU实现:")
        print("   aten/src/ATen/native/SoftMax.cpp")
        print("   - softmax_cpu")
        print("\n3. CUDA实现:")
        print("   aten/src/ATen/native/cuda/SoftMax.cu")
        print("   - softmax_kernel_impl")
        print("   - host_softmax (通用版本)")
        print("   - fast_softmax_kernel (优化版本)")
        print("\n4. Kernel函数:")
        print("   - cunn_SoftMaxForward: 基础实现")
        print("   - SoftMaxForward_kernel: 优化实现")
        print("   - dispatch根据数据大小选择最优kernel")
        
        print("\n【CUDA并行化策略】")
        print("1. 沿归一化维度的并行")
        print("   - 每个block处理一行(或一列)")
        print("   - Grid: (batch_size, seq_len)")
        print("   - Block: 处理整个softmax维度")
        print("\n2. 分阶段计算")
        print("   阶段1: Reduction找最大值")
        print("   - 使用warp shuffle")
        print("   - 或shared memory reduction")
        print("   阶段2: 计算exp和sum")
        print("   - 向量化加载")
        print("   - 融合exp和sum")
        print("   阶段3: 归一化")
        print("   - 并行除法")
        print("\n3. 优化技巧")
        print("   - 在线算法(Online Algorithm): 一次扫描完成")
        print("   - 向量化: 使用float4加载")
        print("   - Warp-level primitives")
        print("   - 融合kernel: 减少内存访问")
        
        print("\n【性能测试】")
        # 创建测试数据
        x = torch.randn(self.batch_size, self.seq_len, self.seq_len).cuda()
        
        # 预热
        self.warm_up(lambda: F.softmax(x, dim=-1))
        
        # 性能测试
        iterations = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            result = F.softmax(x, dim=-1)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_time = start.elapsed_time(end) / iterations
        throughput = (self.batch_size * self.seq_len * self.seq_len * iterations) / (start.elapsed_time(end) / 1000)
        
        print(f"\n输入形状: [{self.batch_size}, {self.seq_len}, {self.seq_len}]")
        print(f"平均时间: {elapsed_time:.4f} ms")
        print(f"吞吐量: {throughput/1e6:.2f} M elements/sec")
        
        return {
            'operation': 'softmax',
            'avg_time_ms': elapsed_time,
            'throughput_m_per_sec': throughput/1e6
        }
    
    def research_layernorm(self):
        """
        调研 layernorm 算子
        """
        print("\n" + "="*80)
        print("3. LAYERNORM 算子调研")
        print("="*80)
        
        print("\n【数学原理】")
        print("LayerNorm 对每个样本的特征进行归一化:")
        print("  y = γ * (x - μ) / √(σ² + ε) + β")
        print("其中:")
        print("  - μ: 均值 = E[x]")
        print("  - σ²: 方差 = E[(x - μ)²]")
        print("  - γ: 可学习的缩放参数")
        print("  - β: 可学习的偏移参数")
        print("  - ε: 防止除零的小常数")
        print("\n应用场景:")
        print("  - Transformer的归一化层")
        print("  - 稳定训练")
        print("  - 加速收敛")
        
        print("\n【PyTorch实现路径】")
        print("Python层:")
        print("  torch.nn.LayerNorm")
        print("    ↓")
        print("C++层:")
        print("  at::layer_norm")
        print("    ↓")
        print("ATen层:")
        print("  aten/src/ATen/native/layer_norm.cpp")
        print("  aten/src/ATen/native/cuda/layer_norm_kernel.cu")
        print("    ↓")
        print("CUDA Kernel:")
        print("  LayerNormForwardKernel")
        
        print("\n【源码位置】")
        print("1. 算子声明:")
        print("   aten/src/ATen/native/native_functions.yaml")
        print("\n2. CUDA实现:")
        print("   aten/src/ATen/native/cuda/layer_norm_kernel.cu")
        print("   - LayerNormForwardKernel")
        print("   - RowwiseMomentsCUDAKernel")
        
        print("\n【CUDA并行化策略】")
        print("详见 layernorm_research.py 的详细分析")
        
        # 性能测试
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size).cuda()
        ln = nn.LayerNorm(self.hidden_size).cuda()
        
        self.warm_up(lambda: ln(x))
        
        iterations = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            result = ln(x)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_time = start.elapsed_time(end) / iterations
        
        print(f"\n输入形状: [{self.batch_size}, {self.seq_len}, {self.hidden_size}]")
        print(f"平均时间: {elapsed_time:.4f} ms")
        
        return {
            'operation': 'layernorm',
            'avg_time_ms': elapsed_time
        }
    
    def research_transpose(self):
        """
        调研 transpose 算子
        """
        print("\n" + "="*80)
        print("4. TRANSPOSE 算子调研")
        print("="*80)
        
        print("\n【数学原理】")
        print("Transpose 交换张量的维度:")
        print("  y[..., i, ..., j, ...] = x[..., j, ..., i, ...]")
        print("\n常见操作:")
        print("  - 矩阵转置: [m, n] → [n, m]")
        print("  - 维度重排: permute/transpose")
        print("  - 内存布局转换")
        print("\n应用场景:")
        print("  - 矩阵运算前的预处理")
        print("  - Attention中的QKV重排")
        print("  - 卷积中的NCHW ↔ NHWC转换")
        
        print("\n【PyTorch实现路径】")
        print("Python层:")
        print("  tensor.transpose(dim0, dim1)")
        print("  tensor.permute(*dims)")
        print("  tensor.t()  # 2D转置")
        print("    ↓")
        print("C++层:")
        print("  at::transpose")
        print("  at::permute")
        print("    ↓")
        print("ATen层:")
        print("  aten/src/ATen/native/TensorTransformations.cpp")
        print("  aten/src/ATen/native/cuda/Copy.cu")
        print("    ↓")
        print("CUDA Kernel:")
        print("  copy_kernel (实际复制数据时)")
        print("  或 view操作 (只改变stride)")
        
        print("\n【源码位置】")
        print("1. 算子声明:")
        print("   aten/src/ATen/native/native_functions.yaml")
        print("   - func: transpose.int(Tensor(a) self, int dim0, int dim1)")
        print("   - func: permute(Tensor(a) self, int[] dims)")
        print("\n2. CPU实现:")
        print("   aten/src/ATen/native/TensorTransformations.cpp")
        print("   - transpose (视图操作)")
        print("   - permute")
        print("\n3. CUDA实现:")
        print("   aten/src/ATen/native/cuda/Copy.cu")
        print("   - copy_device_to_device (需要实际复制时)")
        print("   aten/src/ATen/native/cuda/Transpose.cu")
        print("   - transpose_copy_kernel")
        print("\n4. 关键概念:")
        print("   - Contiguous: 内存连续")
        print("   - View操作: 只改变元数据(stride, shape)")
        print("   - Copy操作: 实际移动数据")
        
        print("\n【CUDA并行化策略】")
        print("1. 视图操作 (View)")
        print("   - 不移动数据,只改变stride和shape")
        print("   - O(1)时间复杂度")
        print("   - 不需要CUDA kernel")
        print("\n2. 复制操作 (Copy)")
        print("   - 需要实际移动数据")
        print("   - 使用transpose_copy_kernel")
        print("\n3. Transpose Copy的优化")
        print("   问题: 非合并访问(Non-coalesced access)")
        print("   - 转置导致stride不连续")
        print("   - 内存访问效率低")
        print("   \n   解决方案: Shared Memory Transpose")
        print("   - 使用shared memory作为中转")
        print("   - 避免bank conflict:")
        print("     __shared__ float tile[TILE_DIM][TILE_DIM + 1];")
        print("     (+1 padding避免bank conflict)")
        print("   \n   算法:")
        print("   步骤1: 合并读取到shared memory")
        print("   步骤2: 在shared memory中转置")
        print("   步骤3: 合并写出")
        print("\n4. 并行配置")
        print("   - Grid: (width/TILE_DIM, height/TILE_DIM)")
        print("   - Block: (TILE_DIM, TILE_DIM)")
        print("   - 典型TILE_DIM = 32")
        
        print("\n【性能测试】")
        # 创建测试数据
        x = torch.randn(1024, 1024).cuda()
        
        # 测试1: View操作(transpose)
        self.warm_up(lambda: x.t())
        
        iterations = 1000
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            y = x.t()
        end.record()
        torch.cuda.synchronize()
        
        view_time = start.elapsed_time(end) / iterations
        
        # 测试2: Copy操作(transpose + contiguous)
        self.warm_up(lambda: x.t().contiguous())
        
        start.record()
        for _ in range(iterations):
            y = x.t().contiguous()
        end.record()
        torch.cuda.synchronize()
        
        copy_time = start.elapsed_time(end) / iterations
        
        print(f"\n矩阵尺寸: [1024, 1024]")
        print(f"View操作时间: {view_time:.6f} ms (几乎为0)")
        print(f"Copy操作时间: {copy_time:.4f} ms")
        print(f"带宽: {1024*1024*4*2 / (copy_time/1000) / 1e9:.2f} GB/s")
        
        return {
            'operation': 'transpose',
            'view_time_ms': view_time,
            'copy_time_ms': copy_time
        }
    
    def research_cuda_runtime(self):
        """
        调研CUDA Runtime API的使用
        """
        print("\n" + "="*80)
        print("5. CUDA RUNTIME API 调研")
        print("="*80)
        
        print("\n【PyTorch中使用的主要CUDA Runtime API】")
        
        print("\n━━━ 1. 内存管理 API ━━━")
        print("\n1.1 内存分配:")
        print("  - cudaMalloc(void** ptr, size_t size)")
        print("    作用: 在GPU上分配内存")
        print("    调用位置: c10/cuda/CUDACachingAllocator.cpp")
        print("    使用场景: 创建tensor、分配中间结果")
        print("\n  - cudaMallocHost(void** ptr, size_t size)")
        print("    作用: 分配pinned memory(页锁定内存)")
        print("    优点: 加速CPU-GPU数据传输")
        print("    调用位置: c10/cuda/CUDACachingAllocator.cpp")
        print("\n  - cudaFree(void* ptr)")
        print("    作用: 释放GPU内存")
        print("    调用位置: c10/cuda/CUDACachingAllocator.cpp")
        
        print("\n1.2 内存拷贝:")
        print("  - cudaMemcpy(dst, src, size, kind)")
        print("    作用: 在CPU和GPU之间拷贝数据")
        print("    kind参数:")
        print("      - cudaMemcpyHostToDevice: CPU → GPU")
        print("      - cudaMemcpyDeviceToHost: GPU → CPU")
        print("      - cudaMemcpyDeviceToDevice: GPU → GPU")
        print("    调用位置: c10/cuda/CUDAStream.cpp")
        print("\n  - cudaMemcpyAsync(dst, src, size, kind, stream)")
        print("    作用: 异步内存拷贝")
        print("    优点: 不阻塞CPU,可与计算重叠")
        print("    调用位置: c10/cuda/CUDAStream.cpp")
        print("\n  - cudaMemset(ptr, value, size)")
        print("    作用: 设置GPU内存的值")
        print("    调用位置: aten/src/ATen/cuda/CUDAContext.cpp")
        
        print("\n━━━ 2. Kernel启动 API ━━━")
        print("\n  - cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream)")
        print("    作用: 启动CUDA kernel")
        print("    参数:")
        print("      - func: kernel函数指针")
        print("      - gridDim: Grid维度(blocks数量)")
        print("      - blockDim: Block维度(threads数量)")
        print("      - args: kernel参数")
        print("      - sharedMem: shared memory大小")
        print("      - stream: CUDA stream")
        print("    调用位置: c10/cuda/CUDAStream.h")
        print("\n  - cudaConfigureCall(gridDim, blockDim, sharedMem, stream)")
        print("    作用: 配置下一次kernel启动(已废弃)")
        print("    现代用法: 使用<<<>>>语法或cudaLaunchKernel")
        
        print("\n━━━ 3. 同步 API ━━━")
        print("\n  - cudaDeviceSynchronize()")
        print("    作用: 等待GPU上所有操作完成")
        print("    特点: 阻塞CPU,等待所有stream")
        print("    调用位置: c10/cuda/CUDAFunctions.cpp")
        print("    使用场景: 计时、调试、确保结果完成")
        print("\n  - cudaStreamSynchronize(stream)")
        print("    作用: 等待指定stream的操作完成")
        print("    特点: 只等待一个stream")
        print("    调用位置: c10/cuda/CUDAStream.cpp")
        print("\n  - cudaEventSynchronize(event)")
        print("    作用: 等待event完成")
        print("    调用位置: c10/cuda/CUDAEvent.h")
        
        print("\n━━━ 4. Stream管理 API ━━━")
        print("\n  - cudaStreamCreate(stream*)")
        print("    作用: 创建CUDA stream")
        print("    用途: 并发执行多个操作")
        print("    调用位置: c10/cuda/CUDAStream.cpp")
        print("\n  - cudaStreamDestroy(stream)")
        print("    作用: 销毁stream")
        print("    调用位置: c10/cuda/CUDAStream.cpp")
        print("\n  - cudaStreamWaitEvent(stream, event)")
        print("    作用: 让stream等待event")
        print("    用途: stream间同步")
        print("    调用位置: c10/cuda/CUDAStream.cpp")
        print("\nStream的作用:")
        print("  - 并发执行: 多个stream可以并发运行")
        print("  - 异步操作: kernel和memcpy可以异步")
        print("  - 隐藏延迟: 计算和传输重叠")
        
        print("\n━━━ 5. Event管理 API ━━━")
        print("\n  - cudaEventCreate(event*)")
        print("    作用: 创建event")
        print("    调用位置: c10/cuda/CUDAEvent.h")
        print("\n  - cudaEventRecord(event, stream)")
        print("    作用: 在stream中记录event")
        print("    用途: 标记时间点")
        print("    调用位置: c10/cuda/CUDAEvent.h")
        print("\n  - cudaEventElapsedTime(ms*, start_event, end_event)")
        print("    作用: 计算两个event之间的时间")
        print("    用途: GPU计时")
        print("    调用位置: c10/cuda/CUDAEvent.h")
        print("\nEvent的作用:")
        print("  - GPU计时: 精确测量kernel执行时间")
        print("  - 同步点: 标记操作完成")
        print("  - Stream同步: 协调多个stream")
        
        print("\n━━━ 6. 设备管理 API ━━━")
        print("\n  - cudaGetDeviceCount(count*)")
        print("    作用: 获取GPU数量")
        print("    调用位置: c10/cuda/CUDAFunctions.cpp")
        print("\n  - cudaSetDevice(device)")
        print("    作用: 设置当前使用的GPU")
        print("    调用位置: c10/cuda/CUDAFunctions.cpp")
        print("\n  - cudaGetDevice(device*)")
        print("    作用: 获取当前GPU ID")
        print("    调用位置: c10/cuda/CUDAFunctions.cpp")
        print("\n  - cudaGetDeviceProperties(prop*, device)")
        print("    作用: 获取GPU属性")
        print("    返回信息:")
        print("      - 名称、计算能力")
        print("      - 显存大小")
        print("      - SM数量、warp大小")
        print("      - shared memory大小")
        print("    调用位置: c10/cuda/CUDAFunctions.cpp")
        
        print("\n━━━ 7. 错误处理 API ━━━")
        print("\n  - cudaGetLastError()")
        print("    作用: 获取最后一个错误")
        print("    返回: cudaError_t")
        print("\n  - cudaPeekAtLastError()")
        print("    作用: 查看错误但不清除")
        print("\n  - cudaGetErrorString(error)")
        print("    作用: 将错误码转为字符串")
        print("\n错误处理宏:")
        print("  PyTorch定义的宏:")
        print("  - C10_CUDA_CHECK(expr)")
        print("  - TORCH_CUDA_CHECK(expr)")
        print("  - AT_CUDA_CHECK(expr)")
        
        print("\n【调用时机和场景】")
        print("\n1. Tensor创建时:")
        print("   - cudaMalloc: 分配GPU内存")
        print("   - cudaGetDevice: 确定在哪个GPU上")
        print("\n2. 数据传输时:")
        print("   - cudaMemcpy / cudaMemcpyAsync")
        print("   - tensor.to(device) 或 tensor.cuda()")
        print("\n3. 算子执行时:")
        print("   - cudaLaunchKernel: 启动kernel")
        print("   - 使用当前stream")
        print("\n4. 性能测试时:")
        print("   - cudaEventCreate/Record/ElapsedTime: 计时")
        print("   - cudaDeviceSynchronize: 等待完成")
        print("\n5. 多GPU训练时:")
        print("   - cudaSetDevice: 切换GPU")
        print("   - cudaStreamCreate: 创建多个stream")
        print("\n6. 释放资源时:")
        print("   - cudaFree: 释放内存")
        print("   - cudaStreamDestroy: 销毁stream")
        
        print("\n【PyTorch的封装】")
        print("\nPyTorch不直接调用CUDA API,而是通过c10库封装:")
        print("\n1. 内存管理:")
        print("   c10::cuda::CUDACachingAllocator")
        print("   - 内存池机制")
        print("   - 减少cudaMalloc调用")
        print("   - 自动管理生命周期")
        print("\n2. Stream管理:")
        print("   c10::cuda::CUDAStream")
        print("   - Stream池")
        print("   - 自动选择stream")
        print("   - RAII风格管理")
        print("\n3. Event管理:")
        print("   c10::cuda::CUDAEvent")
        print("   - 自动创建和销毁")
        print("   - 计时接口简化")
        
        print("\n【示例代码路径】")
        print("\n查看这些文件了解详细使用:")
        print("1. c10/cuda/CUDACachingAllocator.cpp")
        print("   - malloc/free的封装")
        print("2. c10/cuda/CUDAStream.h/cpp")
        print("   - Stream管理")
        print("3. c10/cuda/CUDAEvent.h")
        print("   - Event和计时")
        print("4. c10/cuda/CUDAFunctions.h/cpp")
        print("   - 设备管理函数")
        print("5. aten/src/ATen/cuda/CUDAContext.h/cpp")
        print("   - CUDA上下文管理")
        
        # 演示实际的API使用
        print("\n【实际使用演示】")
        if torch.cuda.is_available():
            print("\n当前GPU信息:")
            print(f"  设备数量: {torch.cuda.device_count()}")
            print(f"  当前设备: {torch.cuda.current_device()}")
            print(f"  设备名称: {torch.cuda.get_device_name()}")
            
            props = torch.cuda.get_device_properties(0)
            print(f"\n设备属性:")
            print(f"  总显存: {props.total_memory / 1e9:.2f} GB")
            print(f"  SM数量: {props.multi_processor_count}")
            
            # 处理不同PyTorch版本的属性名称差异
            try:
                print(f"  每个SM最大线程数: {props.max_threads_per_multi_processor}")
            except AttributeError:
                pass
            
            try:
                print(f"  每个Block最大线程数: {props.max_threads_per_block}")
            except AttributeError:
                print(f"  每个Block最大线程数: 1024 (default)")
            
            # shared_memory_per_block 可能不存在，尝试其他属性名
            try:
                print(f"  Shared Memory (每block): {props.shared_memory_per_block / 1024:.0f} KB")
            except AttributeError:
                try:
                    # 有些版本可能使用不同的名称
                    if hasattr(props, 'total_memory'):
                        print(f"  Shared Memory: 属性不可用")
                except:
                    pass
            
            print(f"  计算能力: {props.major}.{props.minor}")
            
            # 演示Stream使用
            print("\n演示Stream使用:")
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            print(f"  创建了两个Stream: {s1}, {s2}")
            
            # 演示Event计时
            print("\n演示Event计时:")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            x = torch.randn(1000, 1000).cuda()
            start.record()
            y = x @ x.t()
            end.record()
            torch.cuda.synchronize()
            
            print(f"  矩阵乘法时间: {start.elapsed_time(end):.4f} ms")
    
    def profile_all_operators(self):
        """
        使用Profiler分析所有算子
        """
        print("\n" + "="*80)
        print("6. 使用 PyTorch Profiler 分析所有算子")
        print("="*80)
        
        if not torch.cuda.is_available():
            print("CUDA不可用，跳过profiling")
            return
        
        # 准备数据
        batch = 32
        seq = 128
        hidden = 768
        
        x = torch.randn(batch, seq, hidden).cuda()
        weight = torch.randn(hidden, hidden).cuda()
        bias = torch.randn(hidden).cuda()
        ln = nn.LayerNorm(hidden).cuda()
        
        print("\n开始Profiling...")
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(10):
                # addmm
                y1 = torch.addmm(bias, x.view(-1, hidden), weight)
                
                # softmax
                y2 = F.softmax(x, dim=-1)
                
                # layernorm
                y3 = ln(x)
                
                # transpose
                y4 = x.transpose(1, 2)
                y5 = y4.contiguous()
                
                torch.cuda.synchronize()
        
        print("\n各算子的Profiling结果:")
        print("\n--- 按CUDA时间排序 (前30个) ---")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30
        ))
        
        # 保存结果
        output_dir = "./profiler_results"
        os.makedirs(output_dir, exist_ok=True)
        
        trace_path = os.path.join(output_dir, "multi_operator_trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"\n已导出trace到: {trace_path}")
        print("可在 chrome://tracing 中查看")
        
        # 统计各算子时间
        print("\n各算子时间统计:")
        events = prof.key_averages()
        
        operator_times = {
            'addmm': 0,
            'softmax': 0,
            'layer_norm': 0,
            'transpose': 0,
            'copy': 0
        }
        
        for event in events:
            name = event.key.lower()
            if 'addmm' in name or 'gemm' in name or 'blas' in name:
                operator_times['addmm'] += event.cuda_time_total
            elif 'softmax' in name:
                operator_times['softmax'] += event.cuda_time_total
            elif 'layer_norm' in name or 'layernorm' in name:
                operator_times['layer_norm'] += event.cuda_time_total
            elif 'transpose' in name:
                operator_times['transpose'] += event.cuda_time_total
            elif 'copy' in name or 'contiguous' in name:
                operator_times['copy'] += event.cuda_time_total
        
        total_time = sum(operator_times.values())
        print(f"\n{'算子':<15} {'时间(us)':<15} {'占比':<10}")
        print("-" * 40)
        for op, time in sorted(operator_times.items(), key=lambda x: -x[1]):
            pct = (time / total_time * 100) if total_time > 0 else 0
            print(f"{op:<15} {time:<15.2f} {pct:<10.2f}%")
    
    def generate_comprehensive_report(self, results):
        """
        生成综合调研报告
        """
        print("\n" + "="*80)
        print("7. 生成综合调研报告")
        print("="*80)
        
        output_path = "./multi_operator_research_report.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PyTorch CUDA 算子综合调研报告\n")
            f.write("="*80 + "\n\n")
            
            f.write("调研算子: addmm, softmax, layernorm, transpose\n\n")
            
            f.write("一、算子概述\n")
            f.write("-" * 40 + "\n\n")
            
            f.write("1. ADDMM (Add Matrix Multiply)\n")
            f.write("   功能: 矩阵乘加 out = β*input + α*(mat1@mat2)\n")
            f.write("   应用: 全连接层、线性变换\n")
            f.write("   实现: 调用cuBLAS库\n\n")
            
            f.write("2. SOFTMAX\n")
            f.write("   功能: 归一化指数函数\n")
            f.write("   应用: Attention权重、分类输出\n")
            f.write("   实现: 自定义CUDA kernel\n\n")
            
            f.write("3. LAYERNORM\n")
            f.write("   功能: 层归一化\n")
            f.write("   应用: Transformer归一化层\n")
            f.write("   实现: 自定义CUDA kernel\n\n")
            
            f.write("4. TRANSPOSE\n")
            f.write("   功能: 维度交换\n")
            f.write("   应用: 矩阵转置、维度重排\n")
            f.write("   实现: View操作或Copy kernel\n\n")
            
            f.write("\n二、PyTorch源码位置\n")
            f.write("-" * 40 + "\n\n")
            
            f.write("算子声明 (统一位置):\n")
            f.write("  aten/src/ATen/native/native_functions.yaml\n\n")
            
            f.write("ADDMM:\n")
            f.write("  - CPU: aten/src/ATen/native/LinearAlgebra.cpp\n")
            f.write("  - CUDA: aten/src/ATen/native/cuda/Blas.cpp\n")
            f.write("  - cuBLAS: aten/src/ATen/cuda/CUDABlas.cpp\n\n")
            
            f.write("SOFTMAX:\n")
            f.write("  - CPU: aten/src/ATen/native/SoftMax.cpp\n")
            f.write("  - CUDA: aten/src/ATen/native/cuda/SoftMax.cu\n\n")
            
            f.write("LAYERNORM:\n")
            f.write("  - CPU: aten/src/ATen/native/layer_norm.cpp\n")
            f.write("  - CUDA: aten/src/ATen/native/cuda/layer_norm_kernel.cu\n\n")
            
            f.write("TRANSPOSE:\n")
            f.write("  - CPU: aten/src/ATen/native/TensorTransformations.cpp\n")
            f.write("  - CUDA: aten/src/ATen/native/cuda/Copy.cu\n")
            f.write("  - CUDA: aten/src/ATen/native/cuda/Transpose.cu\n\n")
            
            f.write("\n三、CUDA Runtime API使用\n")
            f.write("-" * 40 + "\n\n")
            
            f.write("内存管理:\n")
            f.write("  - cudaMalloc/cudaFree: GPU内存分配释放\n")
            f.write("  - cudaMemcpy: CPU-GPU数据传输\n")
            f.write("  位置: c10/cuda/CUDACachingAllocator.cpp\n\n")
            
            f.write("Kernel启动:\n")
            f.write("  - cudaLaunchKernel: 启动CUDA kernel\n")
            f.write("  位置: c10/cuda/CUDAStream.h\n\n")
            
            f.write("同步操作:\n")
            f.write("  - cudaDeviceSynchronize: 等待所有操作完成\n")
            f.write("  - cudaStreamSynchronize: 等待指定stream\n")
            f.write("  位置: c10/cuda/CUDAFunctions.cpp\n\n")
            
            f.write("Stream管理:\n")
            f.write("  - cudaStreamCreate/Destroy: Stream创建销毁\n")
            f.write("  位置: c10/cuda/CUDAStream.cpp\n\n")
            
            f.write("Event管理:\n")
            f.write("  - cudaEventCreate/Record/ElapsedTime: 计时\n")
            f.write("  位置: c10/cuda/CUDAEvent.h\n\n")
            
            f.write("设备管理:\n")
            f.write("  - cudaSetDevice/GetDevice: GPU选择\n")
            f.write("  - cudaGetDeviceProperties: 获取GPU属性\n")
            f.write("  位置: c10/cuda/CUDAFunctions.cpp\n\n")
            
            f.write("\n四、优化策略总结\n")
            f.write("-" * 40 + "\n\n")
            
            f.write("ADDMM:\n")
            f.write("  - 使用cuBLAS高度优化的GEMM\n")
            f.write("  - Tensor Core加速(支持时)\n")
            f.write("  - 矩阵分块(Tiling)\n\n")
            
            f.write("SOFTMAX:\n")
            f.write("  - 在线算法减少内存访问\n")
            f.write("  - Warp-level primitives\n")
            f.write("  - 向量化加载\n\n")
            
            f.write("LAYERNORM:\n")
            f.write("  - Warp shuffle reduction\n")
            f.write("  - 向量化访问(float4)\n")
            f.write("  - 共享内存优化\n\n")
            
            f.write("TRANSPOSE:\n")
            f.write("  - View操作避免数据移动\n")
            f.write("  - Shared memory避免非合并访问\n")
            f.write("  - Padding避免bank conflict\n\n")
            
            if results:
                f.write("\n五、性能测试结果\n")
                f.write("-" * 40 + "\n\n")
                for result in results:
                    f.write(f"{result}\n")
        
        print(f"报告已保存到: {output_path}")
    
    def run_complete_research(self):
        """
        运行完整调研
        """
        print("\n" + "#"*80)
        print("# 多算子CUDA综合调研")
        print("#"*80)
        
        if torch.cuda.is_available():
            print(f"\nGPU信息:")
            print(f"  设备: {torch.cuda.get_device_name()}")
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        results = []
        
        # 调研各算子
        results.append(self.research_addmm())
        results.append(self.research_softmax())
        results.append(self.research_layernorm())
        results.append(self.research_transpose())
        
        # CUDA Runtime调研
        self.research_cuda_runtime()
        
        # Profiler分析
        self.profile_all_operators()
        
        # 生成报告
        self.generate_comprehensive_report(results)
        
        print("\n" + "#"*80)
        print("# 调研完成!")
        print("#"*80)
        print("\n生成的文件:")
        print("  - profiler_results/multi_operator_trace.json")
        print("  - multi_operator_research_report.txt")


def main():
    """主函数"""
    if not torch.cuda.is_available():
        print("警告: CUDA不可用")
        print("部分功能将无法使用")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    research = MultiOperatorResearch()
    research.run_complete_research()


if __name__ == "__main__":
    main()

