# Transformer模型 CUDA算子优化大作业

## 模型选择：Transformer (BERT)

### 为什么选择Transformer？
1. **应用广泛**：BERT、GPT等模型的基础架构
2. **核心算子明确**：MatMul、Softmax、LayerNorm等
3. **优化空间大**：这些算子都有很大的优化潜力
4. **实际价值高**：当前AI的主流架构

## Transformer核心算子分析

### 1. **矩阵乘法 (GEMM/MatMul)** ⭐⭐⭐⭐⭐
- **作用**：Q、K、V投影，FFN层
- **频率**：非常高（每个attention head + FFN）
- **优化空间**：巨大
- **位置**：
  - `aten/src/ATen/native/cuda/Blas.cpp`
  - `aten/src/ATen/cuda/CUDABlas.cpp`
  - 调用cuBLAS的gemm

### 2. **Softmax** ⭐⭐⭐⭐⭐
- **作用**：Attention权重归一化
- **频率**：每个attention head调用一次
- **优化空间**：中等偏大
- **位置**：
  - `aten/src/ATen/native/cuda/SoftMax.cu`
  - `aten/src/ATen/native/cuda/PersistentSoftmax.cuh`

### 3. **LayerNorm** ⭐⭐⭐⭐⭐
- **作用**：层归一化，stabilize training
- **频率**：每个transformer block调用2次
- **优化空间**：大（reduction操作）
- **位置**：
  - `aten/src/ATen/native/cuda/layer_norm_kernel.cu`

### 4. **Scaled Dot-Product Attention** ⭐⭐⭐⭐⭐
- **作用**：核心的attention计算
- **公式**：softmax(QK^T / sqrt(d_k)) * V
- **优化空间**：巨大（可融合多个操作）
- **位置**：
  - `aten/src/ATen/native/transformers/cuda/attention.cu`

### 5. **GELU激活函数** ⭐⭐⭐
- **作用**：FFN层的激活函数
- **频率**：每个transformer block一次
- **优化空间**：中等
- **位置**：
  - `aten/src/ATen/native/cuda/ActivationGeluKernel.cu`

### 6. **Add (残差连接)** ⭐⭐
- **作用**：残差连接
- **频率**：非常高
- **优化空间**：小（但可融合）
- **位置**：
  - `aten/src/ATen/native/cuda/BinaryOps.cu`

## 推荐调研和优化的3个核心算子

### 方案A：全面覆盖（推荐）
1. **GEMM (矩阵乘法)** - 最核心
2. **Softmax** - attention的关键
3. **LayerNorm** - 归一化操作

### 方案B：深度优化
1. **Fused Attention** - 融合QKV计算和softmax
2. **LayerNorm** 
3. **GELU**

### 方案C：基础扎实
1. **Softmax**
2. **LayerNorm**  
3. **Element-wise Add** (含融合优化)

## 推荐：方案A - 全面覆盖

理由：
- GEMM是计算量最大的操作
- Softmax是attention的核心
- LayerNorm涉及reduction，优化技巧多样
- 三者代表了不同类型的并行化策略

## 各算子的优化方向

### 1. GEMM优化方向
```
原理：C = α * A @ B + β * C

优化技术：
1. Tiling/分块 - 提高数据重用
2. Shared Memory - 减少全局内存访问
3. 向量化访问 - float4等
4. Warp-level优化 - warp shuffle
5. Bank Conflict消除
6. 双缓冲 - 隐藏访存延迟
7. Tensor Core利用（如果硬件支持）
```

**文件位置**：
- 声明：`aten/src/ATen/native/native_functions.yaml` (搜索 `mm`, `matmul`)
- 实现：`aten/src/ATen/cuda/CUDABlas.cpp`
- 会调用cuBLAS，需要自己实现kernel

### 2. Softmax优化方向
```
原理：softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

优化技术：
1. Online Softmax - 单次遍历计算
2. Warp-level reduction - 加速max和sum
3. Persistent kernel - 减少kernel启动开销
4. 数值稳定性处理
5. Vectorized loads/stores
6. Shared memory for reduction
```

**文件位置**：
- 实现：`aten/src/ATen/native/cuda/SoftMax.cu`
- 优化版本：`aten/src/ATen/native/cuda/PersistentSoftmax.cuh`

### 3. LayerNorm优化方向
```
原理：
  mean = sum(x) / N
  var = sum((x - mean)^2) / N
  y = (x - mean) / sqrt(var + eps) * gamma + beta

优化技术：
1. Welford's online algorithm - 单次遍历
2. Warp reduction - 加速均值和方差计算
3. CUB库的reduction原语
4. 向量化读写
5. Fused操作 - 减少内存往返
6. Block-level并行
```

**文件位置**：
- 实现：`aten/src/ATen/native/cuda/layer_norm_kernel.cu`

## 实验设置

### 测试模型
使用标准的BERT-base或BERT-large：
```python
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
model = model.cuda()
model.eval()
```

### 性能测试维度
1. **Batch size**: [1, 4, 8, 16, 32]
2. **Sequence length**: [128, 256, 512]
3. **Hidden size**: 768 (BERT-base), 1024 (BERT-large)

### Profiling步骤
```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        outputs = model(**inputs)

# 查看热点
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 导出Chrome trace
prof.export_chrome_trace("bert_trace.json")
```

## CUDA Runtime API 调研重点

在Transformer中特别关注：

### 1. 内存管理
```cpp
// c10/cuda/CUDACachingAllocator.h
cudaMalloc / cudaFree
- attention中临时tensor的分配
- 缓存机制避免频繁分配
```

### 2. Stream管理
```cpp
// c10/cuda/CUDAStream.h
cudaStreamCreate / cudaStreamSynchronize
- 多层并行执行
- attention计算和FFN的overlap
```

### 3. Kernel启动
```cpp
// 关注grid/block配置
<<<grid_dim, block_dim, shared_mem_size, stream>>>
- GEMM: 2D grid
- Softmax: 1D/2D grid (per-row或per-batch)
- LayerNorm: 1D grid (per-sequence)
```

### 4. 事件同步
```cpp
cudaEventCreate / cudaEventSynchronize
- 精确性能测量
- 不同操作的依赖关系
```

## 目录结构

```
lab1_workspace/
├── 实验计划.md
├── Transformer算子调研计划.md （本文件）
├── 01_安装指南.md
├── 02_算子调研/
│   ├── GEMM分析.md
│   ├── Softmax分析.md
│   ├── LayerNorm分析.md
│   └── CUDA_Runtime_API调研.md
├── 03_profiling/
│   ├── profile_bert.py
│   ├── analyze_kernels.py
│   └── results/
├── 04_implementation/
│   ├── src/
│   │   ├── gemm_optimized.cu
│   │   ├── softmax_optimized.cu
│   │   ├── layernorm_optimized.cu
│   │   └── cuda_utils.cuh
│   ├── test/
│   │   ├── test_gemm.py
│   │   ├── test_softmax.py
│   │   └── test_layernorm.py
│   └── benchmark/
│       ├── benchmark_all.py
│       └── compare_performance.py
├── 05_报告/
│   ├── 实验报告.md
│   ├── figures/
│   └── tables/
└── README.md
```

## 时间规划

### Week 1-2: 环境和Profiling
- [ ] 编译安装PyTorch
- [ ] 运行BERT模型并profiling
- [ ] 识别热点算子

### Week 3-5: 算子调研
- [ ] 深入分析GEMM实现
- [ ] 深入分析Softmax实现
- [ ] 深入分析LayerNorm实现
- [ ] CUDA Runtime API调研

### Week 6-10: 算子优化实现
- [ ] 实现优化版GEMM
- [ ] 实现优化版Softmax
- [ ] 实现优化版LayerNorm
- [ ] 单元测试验证正确性

### Week 11-13: 性能测试
- [ ] Benchmark测试
- [ ] 端到端BERT推理测试
- [ ] 性能分析和可视化

### Week 14-15: 报告撰写
- [ ] 编写详细报告
- [ ] 准备答辩材料

## 参考资源

### 论文
1. "Attention is All You Need" - Transformer原论文
2. "FlashAttention: Fast and Memory-Efficient Exact Attention" - Attention优化
3. "CUTLASS: Fast Linear Algebra in CUDA C++" - GEMM优化

### 代码参考
1. cuBLAS文档
2. CUB库（reduction primitives）
3. FlashAttention实现
4. Apex (NVIDIA的优化库)

### PyTorch源码关键文件
1. `aten/src/ATen/native/cuda/SoftMax.cu`
2. `aten/src/ATen/native/cuda/layer_norm_kernel.cu`
3. `aten/src/ATen/cuda/CUDABlas.cpp`
4. `c10/cuda/CUDAStream.h`
5. `c10/cuda/CUDACachingAllocator.h`

## 评分重点

### 算子调研 (30%)
- ✅ Profiling结果分析
- ✅ 3个算子的详细原理分析
- ✅ 并行化策略说明
- ✅ CUDA Runtime API调研

### 正确性 (25%)
- ✅ 单元测试通过
- ✅ 数值精度验证
- ✅ 端到端模型精度不下降

### 优化效果 (15%)
- ✅ 单算子加速比
- ✅ 端到端加速比
- ✅ 性能分析和解释

### 文档代码 (30%)
- ✅ 清晰的代码注释
- ✅ 详细的实验报告
- ✅ 可复现的README
- ✅ 答辩质量

## 下一步行动

1. **立即开始**：编译安装PyTorch（如果还没有）
2. **第一个脚本**：运行BERT模型的profiling
3. **开始调研**：阅读Softmax和LayerNorm的源码










