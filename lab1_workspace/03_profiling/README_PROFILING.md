# CUDA算子Profiling使用指南（增强版）

## 🎯 目标

使用PyTorch Profiler全面分析Transformer/BERT模型的CUDA算子调用情况，为大作业的算子调研提供数据支持。

## 🚀 快速开始

### 方法1: 使用自动化脚本（推荐）

```bash
cd /home/honglianglu/hdd/CS4302-pytorch/lab1_workspace/03_profiling
./run_comprehensive_profiling.sh
```

### 方法2: 手动运行Python脚本

```bash
# 快速测试（小规模）
python3 profile_bert.py \
    --batch-sizes 1 8 \
    --seq-lens 128 \
    --output-dir ./profiling_results/quick

# 完整测试（多种配置）
python3 profile_bert.py \
    --use-real-bert \
    --batch-sizes 1 4 8 16 \
    --seq-lens 128 256 512 \
    --output-dir ./profiling_results/full
```

## 📊 输出文件说明

每个配置会生成3个文件：

1. **`profiling_stats_bs{N}_seq{M}.json`** - 详细的性能统计数据
   - Top算子列表
   - 算子分类统计
   - Native function映射
   - CUDA kernel信息

2. **`kernel_analysis_report_bs{N}_seq{M}.md`** - Markdown格式分析报告
   - 实验配置
   - 性能总览
   - Top 10算子详细分析
   - CUDA实现文件路径
   - 调研要点建议

3. **`bert_trace_bs{N}_seq{M}.json`** - Chrome Trace可视化文件
   - 在Chrome浏览器中打开 `chrome://tracing`
   - 加载此文件查看时间线

## 📋 查看结果

### 1. 快速查看Top算子

```bash
# 查看Top 3算子
python3 -c "
import json
with open('profiling_results/quick/profiling_stats_bs8_seq128.json') as f:
    data = json.load(f)
    for i, op in enumerate(data['top_aten_operators'][:3], 1):
        print(f'{i}. {op[\"name\"]}')
        print(f'   时间: {op[\"cuda_time_total_ms\"]:.2f}ms')
        print(f'   文件: {op[\"potential_cuda_file\"]}')
        print()
"
```

### 2. 查看分析报告

```bash
ls profiling_results/*/kernel_analysis_report_*.md
cat profiling_results/quick/kernel_analysis_report_bs8_seq128.md
```

### 3. Chrome可视化

1. 打开Chrome浏览器
2. 访问 `chrome://tracing`
3. 点击 "Load" 按钮
4. 选择 `bert_trace_*.json` 文件
5. 使用WASD键导航，鼠标缩放查看

## 🔍 关键算子识别

脚本会自动识别以下类别的算子：

- **Matrix Operations**: mm, matmul, bmm, addmm, gemm
- **Normalization**: layer_norm, batch_norm
- **Activation**: gelu, relu, softmax, sigmoid
- **Attention**: attention, scaled_dot_product
- **Embedding**: embedding, gather
- **Elementwise**: add, mul, div, sub
- **Memory**: copy, clone, transpose, view

## 📝 典型的Top算子（参考）

根据BERT模型特性，通常会发现：

1. **aten::addmm / aten::mm** (40-50%时间)
   - Linear层的矩阵乘法
   - CUDA文件: `aten/src/ATen/native/cuda/Blas.cpp`
   - 调用cuBLAS库

2. **aten::softmax** (10-15%时间)
   - Attention权重计算
   - CUDA文件: `aten/src/ATen/native/cuda/SoftMax.cu`
   - Warp-level reduction

3. **aten::layer_norm** (5-10%时间)
   - Layer Normalization
   - CUDA文件: `aten/src/ATen/native/cuda/layer_norm_kernel.cu`
   - Welford算法

4. **aten::gelu** (3-5%时间)
   - FFN激活函数
   - CUDA文件: `aten/src/ATen/native/cuda/Activation.cu`

5. **aten::bmm** (10-15%时间)
   - Attention中的QK^T和score*V
   - CUDA文件: `aten/src/ATen/native/cuda/Blas.cpp`

## 🎓 用于大作业的调研流程

### Step 1: 收集数据 ✅
```bash
./run_comprehensive_profiling.sh
```

### Step 2: 识别Top 3算子
```bash
# 查看生成的Markdown报告
cat profiling_results/*/kernel_analysis_report_*.md | grep "### 4\.[1-3]"
```

### Step 3: 深入分析CUDA实现

对于每个算子，分析以下内容：

#### 3.1 为何可以并行实现
- 数据独立性分析
- 并行计算的维度

#### 3.2 并行维度的选择
- Block/Thread的组织方式
- Shared memory使用策略
- 寄存器分配

#### 3.3 CUDA Kernel代码逻辑
- 主要计算流程
- 内存访问模式
- 同步点分析

#### 3.4 潜在优化空间
- Memory coalescing
- Bank conflicts避免
- Warp divergence减少
- 算子融合机会

## 📚 源码位置

PyTorch CUDA算子源码：
```
pytorch/aten/src/ATen/native/
├── native_functions.yaml          # 算子声明
├── cuda/
│   ├── SoftMax.cu                # Softmax实现
│   ├── layer_norm_kernel.cu      # LayerNorm实现
│   ├── Activation.cu             # 激活函数
│   ├── Blas.cpp                  # 矩阵运算（调用cuBLAS）
│   └── ...
```

在线查看：https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda

## 🛠️ 高级用法

### 自定义配置
```python
# 修改 profile_bert.py 中的参数
parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 16])
parser.add_argument('--seq-lens', type=int, nargs='+', default=[128, 256])
```

### 使用真实BERT模型
```bash
pip install transformers
python3 profile_bert.py --use-real-bert
```

### 跳过benchmark
```bash
python3 profile_bert.py --skip-benchmark
```

## ❓ 常见问题

### Q1: 如何找到算子对应的native_functions.yaml条目？

查看生成的JSON文件中的 `native_function` 字段，然后在 `native_functions.yaml` 中搜索。

### Q2: CUDA kernel名称太长看不清？

查看JSON文件或使用Chrome trace可视化，可以看到完整名称。

### Q3: 如何对比不同配置的性能？

```python
import json
configs = ['bs1_seq128', 'bs8_seq128', 'bs16_seq128']
for cfg in configs:
    with open(f'profiling_results/full/profiling_stats_{cfg}.json') as f:
        data = json.load(f)
        print(f"{cfg}: {data['summary']['total_cuda_time_ms']:.2f}ms")
```

## 📞 进一步帮助

- PyTorch Profiler文档: https://pytorch.org/docs/stable/profiler.html
- CUDA编程指南: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Nsight Compute: 更详细的kernel级分析工具

---

**最后更新**: 2025-12-02

