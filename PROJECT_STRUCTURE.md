# 项目结构说明

本文档详细说明了项目的目录结构和各文件的作用。

---

## 整体结构

```
lhl/
├── README.md                    # 📖 主文档（必读）
├── QUICKSTART.md               # 🚀 5分钟快速开始指南
├── PROJECT_STRUCTURE.md        # 📁 本文件 - 项目结构说明
│
├── operator_search/            # 📊 模块一：算子性能调研
│   ├── test_new.py            # 🔬 核心测试脚本
│   ├── run_all_benchmarks.sh  # 🔄 批量运行所有算子测试
│   └── output/                # 📈 性能测试结果输出
│       ├── softmax/
│       ├── layernorm/
│       ├── addmm/
│       └── transpose/
│
└── bert_inference_acceleration/ # ⚡ 模块二：融合算子实现
    ├── custom_ops/             # 🎯 CUDA 算子核心实现
    │   ├── custom_gemm.cu     # [967行] CUDA kernel 实现
    │   ├── setup.py           # 编译配置文件
    │   ├── __init__.py        # Python 包初始化
    │   └── *.so               # 编译生成的动态库
    │
    ├── tests/                  # ✅ 测试代码
    │   ├── test_correctness.py    # 正确性验证（必须通过）
    │   └── __init__.py
    │
    ├── benchmarks/             # 📊 性能测试
    │   ├── benchmark.py       # 性能基准测试
    │   └── __init__.py
    │
    ├── examples/               # 💡 使用示例
    │   ├── usage_example.py   # 融合算子使用演示
    │   └── __init__.py
    │
    ├── dataset/                # 📂 测试数据集
    │   ├── imdb/              # IMDB 数据集（电影评论）
    │   └── ag_news/           # AG News 数据集（新闻分类）
    │
    ├── models/                 # 🤖 模型定义（可选）
    │   ├── optimized_bert.py  # 优化的 BERT 模型
    │   └── __init__.py
    │
    ├── data/                   # 🗂️ 数据处理工具
    │   ├── imdb_loader.py     # IMDB 数据加载器
    │   └── __init__.py
    │
    ├── test_multi_dataset_performance.py  # 多数据集性能测试
    ├── test_imdb_performance.py          # IMDB 详细性能测试
    ├── install.sh              # 🛠️ 一键安装脚本
    ├── run_all_tests.sh       # 运行所有测试
    ├── requirements.txt        # Python 依赖列表
    ├── Makefile               # Make 构建配置
    ├── TECHNICAL_EXPLANATION.md  # 技术详解文档
    ├── FINAL_SUMMARY.md       # 项目总结
    └── inference.py           # 推理脚本
```

---

## 📊 模块一：算子性能调研

### 目录: `operator_search/`

**目标：** 通过 PyTorch Profiler 分析 BERT 模型中各算子的性能占比

| 文件 | 作用 | 重要度 |
|------|------|--------|
| `test_new.py` | 核心测试脚本，支持 4 种算子分析 | ⭐⭐⭐⭐⭐ |
| `run_all_benchmarks.sh` | 批量运行脚本，测试所有算子 | ⭐⭐⭐⭐ |
| `output/*/` | 结果输出目录，CSV 格式 | ⭐⭐⭐ |

### 核心文件详解

#### `test_new.py`

**功能：**
- 使用 `torch.profiler` 进行性能分析
- 支持 4 种算子：`softmax`, `layernorm`, `addmm`, `transpose`
- 测试多个 batch size（1, 4, 8, 16, 32, 64, 128）
- 输出 CSV 格式的性能数据

**关键类：**
```python
class BertOperatorResearch:
    def __init__(self, model_name="bert-base-uncased")
    def run_benchmark(self, op_type, dataset_name, num_labels, batch_sizes, seq_len=128)
```

**使用方法：**
```bash
# 测试单个算子
python test_new.py --op addmm

# 支持的算子
python test_new.py --op softmax      # Softmax 算子
python test_new.py --op layernorm    # LayerNorm 算子
python test_new.py --op addmm        # GEMM/矩阵乘法
python test_new.py --op transpose    # 转置操作
```

**输出格式：**
- 文件路径: `output/{op_type}/{dataset}_{op_type}_final.csv`
- 列: BatchSize, TotalTime_us, AbsTime_us, RelTime_%, CUDA_Kernels

#### `run_all_benchmarks.sh`

**功能：**
- 自动化测试所有 4 种算子
- 在 IMDB 和 AG News 两个数据集上测试
- 生成完整的日志文件

**使用方法：**
```bash
bash run_all_benchmarks.sh
```

**日志输出：**
- 实时输出：控制台
- 完整日志：`{BASE_DIR}/benchmark_exec.log`

---

## ⚡ 模块二：融合算子实现

### 目录: `bert_inference_acceleration/`

**目标：** 实现高性能融合算子，减少 BERT 推理延迟

### 核心模块

#### 1. `custom_ops/` - CUDA 算子实现

| 文件 | 行数 | 作用 | 重要度 |
|------|------|------|--------|
| `custom_gemm.cu` | 967 | CUDA kernel 实现 | ⭐⭐⭐⭐⭐ |
| `setup.py` | 30 | 编译配置 | ⭐⭐⭐⭐ |
| `__init__.py` | - | Python 接口 | ⭐⭐⭐ |

**`custom_gemm.cu` 结构：**

```
行 1-16:     头文件和宏定义
行 20-198:   gemm_kernel_optimized<T>      # 高性能 GEMM
行 200-221:  gemm_bias_kernel<T>           # GEMM + Bias
行 223-252:  gemm_bias_gelu_kernel<T>      # GEMM + Bias + GELU
行 254-309:  postprocess_bias_add_layernorm<T>      # 后处理融合
行 311-366:  postprocess_bias_gelu_add_layernorm<T> # 带 GELU 后处理
行 368-476:  gemm_bias_add_layernorm_kernel<T>      # 完整融合（单 kernel）
行 478-584:  gemm_bias_gelu_add_layernorm_kernel<T> # 带 GELU 完整融合
行 586-679:  layernorm_kernel<T>           # LayerNorm
行 681-892:  PyTorch 接口函数              # C++ → Python 绑定
行 894-954:  custom_gemm_bias_gelu_add_layernorm()  # 融合算子接口
行 956-965:  PYBIND11_MODULE               # Python 模块导出
```

**关键函数：**

| 函数名 | 功能 | 对应 PyTorch 操作 |
|--------|------|-------------------|
| `gemm_kernel_optimized` | 高性能矩阵乘法 | `torch.mm` |
| `postprocess_bias_add_layernorm` | Bias+Add+LN 融合 | `+bias`, `+residual`, `LayerNorm` |
| `postprocess_bias_gelu_add_layernorm` | Bias+GELU+Add+LN | `+bias`, `GELU`, `+residual`, `LayerNorm` |
| `custom_gemm_bias_add_layernorm` | 完整融合算子 1 | 5 个操作 → 1 个 |
| `custom_gemm_bias_gelu_add_layernorm` | 完整融合算子 2 | 6 个操作 → 1 个 |

**编译要求（`setup.py`）：**
```python
- CUDA Compute Capability: 7.0, 7.5, 8.0, 8.6
- 编译优化: -O3, --use_fast_math
- 架构支持: V100, RTX 2080Ti, A100, RTX 3090
```

#### 2. `tests/` - 正确性测试

| 文件 | 作用 | 通过标准 |
|------|------|---------|
| `test_correctness.py` | 算子正确性验证 | L2 相对误差 < 1e-4 |

**测试内容：**
1. ✅ GEMM 正确性（3 种尺寸）
2. ✅ GEMM+Bias+GELU 正确性
3. ✅ LayerNorm 正确性

**运行方法：**
```bash
python tests/test_correctness.py
```

**预期输出：**
```
✅ 所有针对 BERT 场景的算子验证通过！
```

#### 3. 性能测试脚本

| 文件 | 测试内容 | 数据集 | 运行时间 |
|------|---------|--------|---------|
| `test_multi_dataset_performance.py` | 多数据集快速测试 | IMDB, AG News | ~1分钟 |
| `test_imdb_performance.py` | IMDB 详细测试 | IMDB | ~2分钟 |
| `benchmarks/benchmark.py` | 综合性能测试 | 可配置 | 可变 |

**`test_multi_dataset_performance.py` 输出：**
```
📊 性能总结报告
数据集    场景         平均长度  PyTorch(ms)  自定义算子(ms)  加速比
IMDB     Attn-Out    277       1.078        1.125          0.96x
IMDB     FFN-Layer   277       3.270        3.890          0.84x
AG News  Attn-Out    56        0.381        0.462          0.82x
AG News  FFN-Layer   56        1.252        1.649          0.76x
```

**`test_imdb_performance.py` 输出：**
- 平均时间 ± 标准差
- P50、P95、P99 百分位延迟
- Kernel 数量减少统计
- 正确性误差（最大、平均）

#### 4. `examples/` - 使用示例

| 文件 | 内容 |
|------|------|
| `usage_example.py` | 3 个完整的使用示例 |

**示例内容：**
1. 基础算子调用
2. Attention 输出层优化
3. FFN 层（带 GELU）优化

**运行方法：**
```bash
python examples/usage_example.py
```

#### 5. 辅助工具

| 文件 | 作用 |
|------|------|
| `install.sh` | 一键安装脚本 |
| `run_all_tests.sh` | 运行所有测试 |
| `requirements.txt` | Python 依赖列表 |
| `Makefile` | Make 构建配置 |

---

## 📦 依赖关系

### 核心依赖

```
torch==2.1.0          # PyTorch 核心框架（必须此版本）
transformers>=4.20.0  # BERT 模型
datasets>=2.0.0       # 数据集加载
numpy>=1.20.0         # 数值计算
tqdm>=4.60.0          # 进度条
pandas>=1.3.0         # 数据处理（算子调研）
tabulate>=0.8.9       # 表格输出
```

### 系统依赖

```
CUDA Toolkit: 11.8+
GCC: 7.0+
Python: 3.10+
GPU: Compute Capability >= 7.0
```

---

## 🔄 工作流程

### 典型使用流程

```
1. 环境准备
   ├── 安装 PyTorch 2.1.0
   ├── 安装其他依赖
   └── 验证 CUDA 可用

2. 算子调研（可选）
   ├── cd operator_search/
   ├── bash run_all_benchmarks.sh
   └── 查看 output/ 目录结果

3. 融合算子安装
   ├── cd bert_inference_acceleration/
   ├── bash install.sh
   └── 验证安装成功

4. 正确性测试
   └── python tests/test_correctness.py

5. 性能评估
   ├── python test_multi_dataset_performance.py
   └── python test_imdb_performance.py

6. 集成到项目（可选）
   └── 参考 examples/usage_example.py
```

---

## 📝 文档索引

| 文档 | 内容 | 适合人群 |
|------|------|---------|
| **README.md** | 完整文档，包含所有细节 | 所有用户 |
| **QUICKSTART.md** | 5 分钟快速开始 | 快速上手 |
| **PROJECT_STRUCTURE.md** | 本文件，项目结构说明 | 开发者 |
| **TECHNICAL_EXPLANATION.md** | 技术深度解析 | 研究者 |
| **FINAL_SUMMARY.md** | 项目总结 | 评审者 |

---

## 🎯 重要文件快速定位

| 我想... | 查看文件 |
|--------|---------|
| 快速上手 | `QUICKSTART.md` |
| 了解算子调研 | `operator_search/test_new.py` |
| 查看 CUDA 实现 | `custom_ops/custom_gemm.cu` |
| 学习如何使用 | `examples/usage_example.py` |
| 验证正确性 | `tests/test_correctness.py` |
| 测试性能 | `test_multi_dataset_performance.py` |
| 安装配置 | `install.sh` |
| 了解编译选项 | `custom_ops/setup.py` |
| 理解技术细节 | `TECHNICAL_EXPLANATION.md` |

---

## 🔧 开发者指南

### 如何修改 CUDA Kernel

1. 编辑 `custom_ops/custom_gemm.cu`
2. 重新编译：
   ```bash
   cd custom_ops
   rm -rf build dist *.so
   pip install -e . --no-build-isolation
   ```
3. 验证正确性：
   ```bash
   cd ..
   python tests/test_correctness.py
   ```
4. 性能测试：
   ```bash
   python test_multi_dataset_performance.py
   ```

### 如何添加新算子

1. 在 `custom_gemm.cu` 添加 CUDA kernel
2. 在 `custom_gemm.cu` 添加 PyTorch 接口函数
3. 在 `PYBIND11_MODULE` 导出新函数
4. 在 `tests/test_correctness.py` 添加测试
5. 重新编译和测试

### 如何调试

```bash
# 编译时启用调试符号
cd custom_ops
CXXFLAGS="-g" pip install -e . --no-build-isolation

# 使用 cuda-gdb 调试
cuda-gdb python tests/test_correctness.py
```

---

## 📊 代码统计

| 模块 | 文件数 | 代码行数 | 语言 |
|------|--------|---------|------|
| operator_search | 2 | ~150 | Python |
| custom_ops (CUDA) | 1 | 967 | C++/CUDA |
| custom_ops (配置) | 2 | ~50 | Python |
| tests | 1 | 129 | Python |
| benchmarks | 3 | ~500 | Python |
| examples | 1 | ~200 | Python |
| **总计** | **10+** | **~2000** | - |

---

**最后更新**: 2026-01-14
**维护者**: lhl
