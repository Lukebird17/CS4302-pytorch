# 多算子CUDA调研工具 - 快速使用

## 包含的文件

- `multi_operator_research.py` - 核心调研程序
- `多算子调研说明.md` - 详细使用说明

## 调研的算子

1. **addmm** - 矩阵乘加操作
2. **softmax** - Softmax归一化  
3. **layernorm** - 层归一化
4. **transpose** - 矩阵转置

## 快速开始

```bash
python multi_operator_research.py
```

## 输出内容

每个算子会输出：
- ✅ 数学原理和公式
- ✅ PyTorch实现路径（Python → C++ → CUDA）
- ✅ 源码位置（具体文件名）
- ✅ CUDA并行化策略
- ✅ 优化方法
- ✅ 性能测试数据

## CUDA Runtime API 调研

详细分析7大类API：
1. 内存管理 (malloc, free, memcpy)
2. Kernel启动 (launchKernel)
3. 同步操作 (synchronize)
4. Stream管理
5. Event管理（计时）
6. 设备管理
7. 错误处理

每个API都说明：
- 作用
- 在PyTorch源码中的位置
- 使用场景
- 代码示例

## 生成的文件

```
profiler_results/
  └── multi_operator_trace.json     # Chrome trace可视化

multi_operator_research_report.txt  # 综合调研报告
```

## PyTorch源码位置总结

### 算子声明（统一）
```
aten/src/ATen/native/native_functions.yaml
```

### ADDMM
```
aten/src/ATen/native/LinearAlgebra.cpp       # CPU
aten/src/ATen/native/cuda/Blas.cpp           # CUDA
aten/src/ATen/cuda/CUDABlas.cpp              # cuBLAS
```

### SOFTMAX
```
aten/src/ATen/native/SoftMax.cpp             # CPU
aten/src/ATen/native/cuda/SoftMax.cu         # CUDA
```

### LAYERNORM
```
aten/src/ATen/native/layer_norm.cpp          # CPU
aten/src/ATen/native/cuda/layer_norm_kernel.cu # CUDA
```

### TRANSPOSE
```
aten/src/ATen/native/TensorTransformations.cpp # CPU
aten/src/ATen/native/cuda/Copy.cu             # Copy
aten/src/ATen/native/cuda/Transpose.cu        # Transpose
```

### CUDA Runtime封装
```
c10/cuda/CUDACachingAllocator.cpp  # 内存管理
c10/cuda/CUDAStream.h/cpp          # Stream管理
c10/cuda/CUDAEvent.h               # Event和计时
c10/cuda/CUDAFunctions.h/cpp       # 设备管理
```

## 报告撰写建议

### 算子调研部分
每个算子包含：
1. 数学原理
2. PyTorch实现路径
3. 源码位置
4. CUDA并行化策略
5. 优化方法
6. 性能数据

### CUDA Runtime部分
1. API分类和说明
2. 在PyTorch中的调用位置
3. 使用场景和时机
4. 代码示例

### 实验数据部分
1. Profiler结果截图
2. 性能测试数据
3. 时间占比分析

## 详细说明

查看 `多算子调研说明.md` 获取完整的使用说明。

---

**现在就可以运行了！** 🚀

```bash
python multi_operator_research.py
```


