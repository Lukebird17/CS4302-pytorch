# 文档索引

快速找到你需要的文档。

---

## 📚 主要文档

### 1. 🚀 [QUICKSTART.md](QUICKSTART.md) - 快速开始（5分钟）

**适合：** 第一次使用的用户

**内容：**
- ✅ 环境检查（2分钟）
- ✅ 算子调研快速运行（1分钟）
- ✅ 融合算子安装（2分钟）
- ✅ 验证和测试（1分钟）
- ✅ 常见问题速查

**使用场景：**
- 刚接触项目，想快速跑起来
- 时间紧张，需要马上验证环境
- 演示或答辩前的最后检查

---

### 2. 📖 [README.md](README.md) - 完整文档

**适合：** 需要详细了解项目的用户

**内容：**
- 📋 项目结构和目录说明
- 🔧 环境要求和配置详解
- 📊 模块一：算子性能调研（详细）
- 🔥 模块二：融合算子实现（详细）
- 📈 实验结果和性能分析
- 🔬 技术细节和实现原理
- ❓ 常见问题完整列表
- 📚 参考资料

**使用场景：**
- 需要深入理解项目
- 遇到问题需要详细排查
- 想修改或扩展代码
- 准备答辩或报告

---

### 3. 📁 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 项目结构

**适合：** 开发者和维护者

**内容：**
- 📂 完整的目录树和文件说明
- 📊 每个模块的详细介绍
- 🔄 工作流程图
- 🎯 重要文件快速定位表
- 🔧 开发者指南
- 📊 代码统计

**使用场景：**
- 需要了解代码组织结构
- 寻找特定功能的实现位置
- 准备修改或扩展代码
- 代码审查或维护

---

### 4. 🔬 [TECHNICAL_EXPLANATION.md](bert_inference_acceleration/TECHNICAL_EXPLANATION.md) - 技术详解

**适合：** 研究者和算法工程师

**内容：**
- 🧮 CUDA Kernel 实现细节
- ⚡ 优化技术深度解析
  - Tile-based GEMM
  - 双缓冲技术
  - Warp Shuffle Reduction
  - Bank Conflict 避免
- 📊 性能模型和理论分析
- 🔍 与其他实现的对比

**使用场景：**
- 深入理解优化技术
- 学习 CUDA 编程
- 准备技术报告或论文
- 性能调优和进一步优化

---

## 🎯 按使用场景查找

### 场景 1: 我想快速运行起来

**推荐阅读顺序：**
1. [QUICKSTART.md](QUICKSTART.md) - 快速开始
2. 运行测试验证
3. 如有问题，查看 [README.md](README.md) 的"常见问题"部分

**预计时间：** 10-15 分钟

---

### 场景 2: 我要准备答辩/报告

**推荐阅读顺序：**
1. [README.md](README.md) - 了解整体
2. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 理解结构
3. [TECHNICAL_EXPLANATION.md](bert_inference_acceleration/TECHNICAL_EXPLANATION.md) - 掌握技术细节
4. 运行所有测试，截图保存结果
5. 查看 `operator_search/output/` 的性能数据

**关键要点：**
- ✅ 两个主要模块：算子调研 + 融合算子实现
- ✅ PyTorch 版本：2.1.0（重要！）
- ✅ 优化技术：双缓冲、Warp Shuffle、向量化、Bank Conflict 避免
- ✅ 性能结果：内存访问减少 50-60%，Kernel 启动减少 60-70%
- ✅ 正确性保证：L2 相对误差 < 1e-6

**预计时间：** 2-3 小时

---

### 场景 3: 我要修改或扩展代码

**推荐阅读顺序：**
1. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 定位文件
2. [README.md](README.md) - 理解设计思路
3. [TECHNICAL_EXPLANATION.md](bert_inference_acceleration/TECHNICAL_EXPLANATION.md) - 学习实现细节
4. 查看相关源代码
5. 参考 `examples/usage_example.py`

**关键文件：**
- CUDA 实现: `custom_ops/custom_gemm.cu`
- 编译配置: `custom_ops/setup.py`
- 测试代码: `tests/test_correctness.py`
- 性能测试: `test_multi_dataset_performance.py`

**预计时间：** 半天到一天

---

### 场景 4: 我要集成到自己的项目

**推荐阅读：**
1. [QUICKSTART.md](QUICKSTART.md) - 确保环境正确
2. [examples/usage_example.py](bert_inference_acceleration/examples/usage_example.py) - 学习使用方法
3. [README.md](README.md) 的"模块二"部分

**核心代码示例：**
```python
from custom_ops_cuda import gemm_bias_add_layernorm

# 在你的模型中使用
output = gemm_bias_add_layernorm(
    input_tensor,
    weight.t().contiguous(),
    bias,
    residual,
    gamma,
    beta,
    eps
)
```

**预计时间：** 1-2 小时

---

## 📊 按模块查找

### 模块一：算子性能调研 (`operator_search/`)

| 文档章节 | 文档位置 |
|---------|---------|
| 快速运行 | [QUICKSTART.md - 算子调研](QUICKSTART.md#2%EF%B8%8F⃣-算子调研1分钟) |
| 详细说明 | [README.md - 模块一](README.md#📊-模块一算子性能调研) |
| 代码说明 | [PROJECT_STRUCTURE.md - 模块一](PROJECT_STRUCTURE.md#📊-模块一算子性能调研) |

**核心文件：**
- `operator_search/test_new.py` - 测试脚本
- `operator_search/run_all_benchmarks.sh` - 批量运行

---

### 模块二：融合算子实现 (`bert_inference_acceleration/`)

| 文档章节 | 文档位置 |
|---------|---------|
| 快速运行 | [QUICKSTART.md - 融合算子](QUICKSTART.md#3%EF%B8%8F⃣-融合算子安装2分钟) |
| 详细说明 | [README.md - 模块二](README.md#🔥-模块二融合算子实现) |
| 代码说明 | [PROJECT_STRUCTURE.md - 模块二](PROJECT_STRUCTURE.md#⚡-模块二融合算子实现) |
| 技术细节 | [TECHNICAL_EXPLANATION.md](bert_inference_acceleration/TECHNICAL_EXPLANATION.md) |

**核心文件：**
- `custom_ops/custom_gemm.cu` - CUDA 实现（967 行）
- `custom_ops/setup.py` - 编译配置
- `tests/test_correctness.py` - 正确性验证
- `test_multi_dataset_performance.py` - 性能测试

---

## 🔍 常见问题快速索引

| 问题 | 查看 |
|------|------|
| 环境配置问题 | [README.md - 环境要求](README.md#🔧-环境要求) |
| 编译失败 | [README.md - 常见问题 Q1](README.md#q1-编译失败提示找不到-cuda) |
| 导入错误 | [README.md - 常见问题 Q2](README.md#q2-运行时提示-importerror-cannot-import-name-gemm_bias_add_layernorm) |
| PyTorch 版本 | [README.md - 常见问题 Q3](README.md#q3-pytorch-版本不匹配) |
| 数据集下载 | [README.md - 常见问题 Q4](README.md#q4-测试数据集下载失败) |
| 性能不如预期 | [README.md - 常见问题 Q5](README.md#q5-为什么融合算子性能没有显著提升) |
| 如何使用 | [README.md - 常见问题 Q6](README.md#q6-如何在自己的模型中使用融合算子) |

---

## 📝 文件清单

### 根目录 (`lhl/`)

```
✅ README.md                 - 主文档（必读）
✅ QUICKSTART.md            - 快速开始（5分钟）
✅ PROJECT_STRUCTURE.md     - 项目结构说明
✅ DOCUMENTATION_INDEX.md   - 本文件（文档索引）
```

### 算子调研 (`operator_search/`)

```
✅ test_new.py              - 测试脚本
✅ run_all_benchmarks.sh    - 批量运行脚本
📁 output/                  - 结果输出目录
```

### 融合算子 (`bert_inference_acceleration/`)

```
📁 custom_ops/
   ✅ custom_gemm.cu        - CUDA 实现（967 行）
   ✅ setup.py              - 编译配置
   ✅ __init__.py

📁 tests/
   ✅ test_correctness.py   - 正确性测试

📁 examples/
   ✅ usage_example.py      - 使用示例

✅ test_multi_dataset_performance.py  - 多数据集测试
✅ test_imdb_performance.py          - IMDB 详细测试
✅ install.sh                         - 一键安装
✅ requirements.txt                   - 依赖列表
✅ TECHNICAL_EXPLANATION.md          - 技术详解
```

---

## 🎓 学习路径

### 初学者路径

```
第1天：快速上手
├── 阅读 QUICKSTART.md
├── 运行算子调研
├── 安装融合算子
└── 运行正确性测试

第2天：深入理解
├── 阅读 README.md
├── 理解项目结构（PROJECT_STRUCTURE.md）
├── 运行性能测试
└── 查看测试结果

第3天：技术深入
├── 阅读 TECHNICAL_EXPLANATION.md
├── 查看 CUDA 代码（custom_gemm.cu）
├── 学习优化技术
└── 尝试修改参数
```

### 进阶路径

```
Week 1: 理解现有实现
├── 深入研究 CUDA Kernel
├── 分析性能瓶颈
├── 学习优化技术
└── 尝试小的改进

Week 2: 扩展功能
├── 添加新的融合算子
├── 支持更多数据类型（FP16/INT8）
├── 优化特定硬件（Tensor Core）
└── 集成到完整 BERT 模型

Week 3: 性能优化
├── 调优 Tile 大小
├── 优化寄存器使用
├── 改进内存访问模式
└── 实现自动调优
```

---

## 💡 使用建议

1. **第一次使用：** 从 [QUICKSTART.md](QUICKSTART.md) 开始
2. **准备答辩：** 认真阅读 [README.md](README.md) 和 [TECHNICAL_EXPLANATION.md](bert_inference_acceleration/TECHNICAL_EXPLANATION.md)
3. **代码开发：** 参考 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 和 [examples/usage_example.py](bert_inference_acceleration/examples/usage_example.py)
4. **问题排查：** 查看 [README.md](README.md) 的"常见问题"章节
5. **深入研究：** 阅读 [TECHNICAL_EXPLANATION.md](bert_inference_acceleration/TECHNICAL_EXPLANATION.md) 和源代码

---

## 📞 需要帮助？

1. 查看对应文档的"常见问题"部分
2. 检查 [README.md](README.md) 的完整问题列表
3. 查看 GitHub Issues（如果有）
4. 联系项目维护者

---

**文档版本：** v1.0  
**最后更新：** 2026-01-14  
**维护者：** lhl
