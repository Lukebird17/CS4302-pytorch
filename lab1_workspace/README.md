# CS4302 PyTorch CUDA算子优化大作业

## 项目概述

本项目针对Transformer模型（BERT）进行CUDA算子调研和优化，包括：
1. 使用BERT-base-uncased进行文本分类任务（IMDB/AG News）
2. Profiling分析识别热点CUDA kernel
3. 深入调研核心算子的CUDA实现
4. 优化重写关键算子（Softmax、LayerNorm、GEMM等）
5. 性能对比和分析

## 目录结构

```
lab1_workspace/
├── README.md                           # 本文件
├── 实验计划.md                         # 详细的实验计划
├── Transformer算子调研计划.md          # Transformer算子调研方案
├── 01_安装指南.md                      # PyTorch源码编译安装指南
├── quick_start.sh                      # 快速启动脚本
│
├── 02_算子调研/                        # 算子源码分析文档
│   ├── 01_Softmax算子分析.md           # Softmax详细分析
│   ├── 02_LayerNorm算子分析.md         # LayerNorm详细分析
│   ├── 03_GEMM算子分析.md              # 矩阵乘法详细分析
│   └── 04_CUDA_Runtime_API调研.md     # CUDA Runtime API调研
│
├── 03_profiling/                       # Profiling脚本和结果
│   ├── README.md                       # Profiling使用指南
│   ├── profile_bert.py                 # 简化版profiling脚本
│   └── train_bert_classification.py    # 完整训练+profiling脚本
│
├── 04_implementation/                  # 优化算子实现
│   ├── src/                            # 优化后的CUDA源码
│   │   ├── softmax_optimized.cu
│   │   ├── layernorm_optimized.cu
│   │   ├── gemm_optimized.cu
│   │   └── cuda_utils.cuh
│   ├── test/                           # 单元测试
│   │   ├── test_softmax.py
│   │   ├── test_layernorm.py
│   │   └── test_gemm.py
│   └── benchmark/                      # 性能测试
│       ├── benchmark_all.py
│       └── compare_performance.py
│
└── 05_报告/                            # 实验报告和文档
    ├── 实验报告.md
    ├── figures/
    └── tables/
```

## 快速开始

### 方法1: 使用快速启动脚本（推荐）

```bash
cd /home/honglianglu/hdd/CS4302-pytorch/lab1_workspace
./quick_start.sh
```

这将自动：
- ✅ 检查环境配置
- ✅ 安装必要依赖
- ✅ 运行快速测试
- ✅ 生成profiling结果

### 方法2: 手动步骤

#### Step 1: 安装依赖
```bash
pip install torch transformers datasets tqdm
```

#### Step 2: 训练BERT模型
```bash
cd 03_profiling

# IMDB情感分类
python train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --epochs 3 \
    --do_train \
    --do_eval \
    --output_dir ../outputs/imdb
```

#### Step 3: Profiling分析
```bash
python train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --do_profile \
    --load_checkpoint ../outputs/imdb/best_model \
    --output_dir ../outputs/imdb
```

## 实验流程

### 阶段1: 环境准备（第1-2周）
- [x] 确认Python、CUDA、GCC版本
- [ ] 编译安装PyTorch源码（如需修改底层实现）
- [x] 安装transformers、datasets等依赖
- [x] 运行快速测试验证环境

**当前状态**: 
- Python 3.13.9
- CUDA 12.4
- GCC 11.4.0
- PyTorch源码位于: `/home/honglianglu/hdd/CS4302-pytorch`

### 阶段2: 模型训练和Profiling（第3-5周）
- [x] 选择数据集：IMDB（情感分类）或AG News（新闻分类）
- [x] 创建训练脚本
- [x] 训练BERT-base模型
- [ ] 使用PyTorch Profiler分析
- [ ] 生成Chrome trace可视化
- [ ] 识别热点CUDA kernel

**关键文件**:
- 训练脚本: `03_profiling/train_bert_classification.py`
- Profiling指南: `03_profiling/README.md`

### 阶段3: CUDA算子调研（第3-5周）
调研至少3个核心算子，推荐：

#### 1. Softmax ⭐⭐⭐⭐⭐
- **源码位置**: `aten/src/ATen/native/cuda/SoftMax.cu`
- **调研重点**:
  - Warp-level reduction实现
  - Persistent kernel策略
  - 数值稳定性处理
  - Online softmax算法
- **文档**: `02_算子调研/01_Softmax算子分析.md` ✅

#### 2. LayerNorm ⭐⭐⭐⭐⭐
- **源码位置**: `aten/src/ATen/native/cuda/layer_norm_kernel.cu`
- **调研重点**:
  - Welford online算法
  - Block/Warp reduction
  - 向量化访存
  - Fused kernel设计
- **文档**: `02_算子调研/02_LayerNorm算子分析.md`

#### 3. GEMM (矩阵乘法) ⭐⭐⭐⭐⭐
- **源码位置**: `aten/src/ATen/cuda/CUDABlas.cpp`
- **调研重点**:
  - Tiling/分块策略
  - Shared memory优化
  - Warp-level matmul
  - Tensor Core利用
- **文档**: `02_算子调研/03_GEMM算子分析.md`

#### 4. CUDA Runtime API 调研
- **源码位置**: `c10/cuda/`目录
- **调研重点**:
  - 内存管理: cudaMalloc/cudaFree
  - Stream管理: cudaStreamCreate
  - Kernel启动配置: <<<grid, block>>>
  - 事件和同步: cudaEventCreate
- **文档**: `02_算子调研/04_CUDA_Runtime_API调研.md`

### 阶段4: 算子优化实现（第6-10周）

#### 实现要求
- ✅ 选择关键算子进行优化（至少1个）
- ⚠️ 不要直接调用cuBLAS库（针对GEMM）
- ✅ 实现正确性验证测试
- ✅ 实现性能benchmark

#### 优化技术清单
```
□ Shared Memory优化
□ 向量化访存（float4）
□ Bank Conflict消除
□ Warp Shuffle
□ 循环展开
□ Persistent Threads
□ Occupancy优化
□ Memory Coalescing
□ Double Buffering
□ Kernel Fusion
```

#### 实现文件组织
```
04_implementation/
├── src/
│   ├── softmax_optimized.cu       # 优化的Softmax
│   ├── layernorm_optimized.cu     # 优化的LayerNorm
│   ├── gemm_optimized.cu          # 优化的GEMM（不用cuBLAS）
│   └── cuda_utils.cuh             # 通用CUDA工具
├── test/
│   └── test_*.py                  # 正确性测试
└── benchmark/
    └── benchmark_all.py           # 性能测试
```

### 阶段5: 性能测试（第11-13周）

#### 测试维度
- Batch size: [1, 4, 8, 16, 32]
- Sequence length: [128, 256, 512]
- 不同GPU型号

#### 评估指标
- 单算子加速比
- 端到端推理加速比
- 内存带宽利用率
- GPU占用率

#### 对比基准
- PyTorch官方实现
- cuDNN/cuBLAS（如果适用）

### 阶段6: 报告撰写（第14-15周）

#### 报告结构
```
1. 摘要与分工
2. 实验环境配置
3. CUDA算子调研
   3.1 Profiling结果分析
   3.2 算子原理和实现分析
   3.3 CUDA Runtime API调研
4. 算子优化实现
   4.1 优化方法设计
   4.2 代码实现详解
   4.3 关键优化技术
5. 实验结果
   5.1 正确性验证
   5.2 性能对比数据
   5.3 性能分析
6. 结论与心得
7. 参考资料
```

#### 提交清单
- [ ] 实验报告PDF
- [ ] 完整代码（带注释）
- [ ] README（包含PyTorch版本和文件路径）
- [ ] 性能测试结果和截图
- [ ] 可复现的运行脚本

### 阶段7: 答辩准备（第12周和第16周）

#### 中期答辩（第12周）
- 算子调研结果
- 优化方案设计
- 初步实现进展

#### 最终答辩（第16周）
- 完整的优化实现
- 性能对比结果
- Demo演示
- 问题回答

## 重要文件索引

### 配置和指南
- `实验计划.md`: 完整的实验计划和时间线
- `Transformer算子调研计划.md`: 算子选择和优化方向
- `01_安装指南.md`: PyTorch编译安装
- `quick_start.sh`: 快速启动脚本

### 训练和测试
- `03_profiling/train_bert_classification.py`: BERT训练+profiling
- `03_profiling/profile_bert.py`: 简化版profiling
- `03_profiling/README.md`: 使用说明

### 算子调研
- `02_算子调研/01_Softmax算子分析.md`: Softmax详细分析 ✅
- `02_算子调研/02_LayerNorm算子分析.md`: LayerNorm分析
- `02_算子调研/03_GEMM算子分析.md`: GEMM分析
- `02_算子调研/04_CUDA_Runtime_API调研.md`: Runtime API

### PyTorch源码关键文件
```
/home/honglianglu/hdd/CS4302-pytorch/
├── aten/src/ATen/native/
│   ├── native_functions.yaml              # 算子声明
│   └── cuda/
│       ├── SoftMax.cu                     # Softmax实现
│       ├── layer_norm_kernel.cu           # LayerNorm实现
│       └── ...
├── aten/src/ATen/cuda/
│   └── CUDABlas.cpp                       # BLAS封装
└── c10/cuda/
    ├── CUDAStream.h                       # Stream管理
    └── CUDACachingAllocator.h             # 内存管理
```

## 常用命令

### Profiling
```bash
# 完整的训练+profiling
python 03_profiling/train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --do_train --do_eval --do_profile \
    --output_dir outputs/imdb

# 只做profiling（使用已训练模型）
python 03_profiling/train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --do_profile \
    --load_checkpoint outputs/imdb/best_model \
    --output_dir outputs/imdb
```

### 查看结果
```bash
# 查看kernel统计
cat outputs/imdb/profiling_results/kernel_statistics.json | jq

# 在Chrome中查看trace
# 1. 打开 chrome://tracing
# 2. 加载 outputs/imdb/profiling_results/bert_inference_trace.json
```

### 查找源码
```bash
cd /home/honglianglu/hdd/CS4302-pytorch

# 查找Softmax相关文件
find aten/src/ATen/native/cuda -name "*softmax*" -o -name "*Softmax*"

# 查找LayerNorm相关文件
find aten/src/ATen/native/cuda -name "*layer_norm*" -o -name "*LayerNorm*"

# 搜索特定函数
grep -r "softmax_warp_forward" aten/src/ATen/native/cuda/
```

## 参考资料

### 论文
1. "Attention is All You Need" - Transformer原论文
2. "FlashAttention: Fast and Memory-Efficient Exact Attention"
3. "FlashAttention-2: Faster Attention with Better Parallelism"
4. CUTLASS: Fast Linear Algebra in CUDA C++

### 文档
1. [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
2. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
3. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 代码参考
1. PyTorch源码: `/home/honglianglu/hdd/CS4302-pytorch`
2. [FlashAttention](https://github.com/Dao-AILab/flash-attention)
3. [CUTLASS](https://github.com/NVIDIA/cutlass)
4. [NVIDIA Apex](https://github.com/NVIDIA/apex)

## 评分标准

| 项目 | 分值 | 要求 |
|------|------|------|
| 算子调研 | 30% | Profiling分析 + 3个算子详细分析 + Runtime API |
| 正确性 | 25% | 单元测试 + 精度验证 |
| 优化效果 | 15% | 加速比 + 性能分析 |
| 文档代码 | 30% | 报告质量 + 代码质量 + 答辩 |

## 常见问题

### Q: 需要重新编译PyTorch吗？
A: 如果只是调研和profiling，不需要。如果要修改底层算子实现，需要编译。参考`01_安装指南.md`。

### Q: IMDB和AG News选哪个？
A: 推荐IMDB，数据集较小（25K训练样本），训练更快。AG News更大（120K样本）但分类任务更简单。

### Q: 显存不够怎么办？
A: 减小batch_size或max_length。最小可以用batch_size=1, max_length=128。

### Q: 如何验证优化算子的正确性？
A: 与PyTorch原始实现对比输出，使用`torch.allclose()`检查数值误差在可接受范围（rtol=1e-5）。

### Q: GEMM一定要自己实现吗？
A: 作业要求"不要直接调用cuBLAS库"，所以如果选择优化GEMM，需要自己实现基本的矩阵乘法kernel。

## 联系方式

如有问题，参考：
1. 本README和相关文档
2. PyTorch官方文档
3. CUDA编程指南
4. 课程讨论区

---

**最后更新**: 2025-12-01
**PyTorch版本**: Main branch (commit: 7bcf7da3a2)
**工作目录**: `/home/honglianglu/hdd/CS4302-pytorch/lab1_workspace`










