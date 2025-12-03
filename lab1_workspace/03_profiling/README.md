# BERT文本分类 Profiling 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install transformers datasets torch tqdm
```

### 2. 训练BERT模型

#### IMDB情感分类（推荐先用这个，数据集较小）
```bash
# 训练模型
python train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --epochs 3 \
    --do_train \
    --do_eval \
    --output_dir ./outputs/imdb

# 只评估（如果已经有训练好的模型）
python train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --do_eval \
    --load_checkpoint ./outputs/imdb/best_model \
    --output_dir ./outputs/imdb
```

#### AG News新闻分类
```bash
python train_bert_classification.py \
    --dataset agnews \
    --batch_size 16 \
    --epochs 3 \
    --do_train \
    --do_eval \
    --output_dir ./outputs/agnews
```

### 3. Profiling分析

```bash
# 对训练好的模型进行profiling
python train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --do_profile \
    --load_checkpoint ./outputs/imdb/best_model \
    --output_dir ./outputs/imdb

# 会生成以下文件：
# - profiling_results/bert_inference_trace.json  (Chrome trace可视化)
# - profiling_results/kernel_statistics.json     (Kernel统计数据)
# - profiling_results/benchmark_stats.json       (性能benchmark)
```

### 4. 查看Profiling结果

#### 方法1: 查看终端输出
运行脚本后会直接在终端打印Top 30的CUDA kernel统计。

#### 方法2: Chrome Trace可视化
1. 打开Chrome浏览器
2. 访问 `chrome://tracing`
3. 点击"Load"按钮
4. 选择生成的 `bert_inference_trace.json` 文件
5. 可以交互式地查看各个kernel的时间线

#### 方法3: 查看JSON统计文件
```bash
# 查看kernel统计
cat outputs/imdb/profiling_results/kernel_statistics.json

# 查看benchmark结果
cat outputs/imdb/profiling_results/benchmark_stats.json
```

## 脚本参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | imdb | 数据集选择: imdb 或 agnews |
| `--model_name` | bert-base-uncased | 预训练模型名称 |
| `--max_length` | 128 | 最大序列长度 |
| `--batch_size` | 16 | 批次大小 |
| `--epochs` | 3 | 训练轮数 |
| `--lr` | 2e-5 | 学习率 |
| `--output_dir` | ./outputs | 输出目录 |
| `--do_train` | False | 是否训练 |
| `--do_eval` | False | 是否评估 |
| `--do_profile` | False | 是否profiling |
| `--load_checkpoint` | None | 加载checkpoint路径 |

## 完整工作流程

### Step 1: 快速测试（使用小数据集）
```bash
# 使用IMDB的子集快速训练和测试
python train_bert_classification.py \
    --dataset imdb \
    --batch_size 8 \
    --epochs 1 \
    --do_train \
    --do_eval \
    --do_profile \
    --output_dir ./outputs/test
```

### Step 2: 完整训练
```bash
# 完整训练IMDB模型
python train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --epochs 3 \
    --do_train \
    --output_dir ./outputs/imdb_full
```

### Step 3: 详细Profiling
```bash
# 对训练好的模型进行详细profiling
python train_bert_classification.py \
    --dataset imdb \
    --batch_size 16 \
    --do_profile \
    --load_checkpoint ./outputs/imdb_full/best_model \
    --output_dir ./outputs/imdb_full
```

### Step 4: 不同配置的性能测试
```bash
# 测试不同batch size的性能
for bs in 1 4 8 16 32; do
    echo "Testing batch_size=$bs"
    python train_bert_classification.py \
        --dataset imdb \
        --batch_size $bs \
        --do_profile \
        --load_checkpoint ./outputs/imdb_full/best_model \
        --output_dir ./outputs/imdb_bs${bs}
done
```

## Profiling输出解读

### 关键算子识别

运行profiling后，你会看到类似这样的输出：

```
算子名称                                                       总时间(ms)      调用次数    平均(ms)
----------------------------------------------------------------------------------------------------
aten::addmm                                                   125.234         240        0.522
aten::softmax                                                 45.678          144        0.317
aten::layer_norm                                              38.456          288        0.134
aten::gelu                                                    15.234          144        0.106
aten::bmm                                                     98.765          288        0.343
```

### 重点关注的算子

基于Transformer架构，重点关注以下算子：

1. **aten::addmm / aten::mm / aten::matmul**
   - 矩阵乘法（GEMM）
   - 用于Q、K、V投影和FFN层
   - 通常占总时间的40-60%

2. **aten::softmax**
   - Softmax归一化
   - 用于attention权重计算
   - 占总时间的5-15%

3. **aten::layer_norm**
   - Layer Normalization
   - 每个transformer block调用2次
   - 占总时间的5-10%

4. **aten::gelu**
   - GELU激活函数
   - FFN层使用
   - 占总时间的2-5%

5. **aten::bmm**
   - Batch矩阵乘法
   - Attention计算中的QK^T和(QK^T)V
   - 占总时间的10-20%

### 性能指标

查看 `benchmark_stats.json` 了解：
- **吞吐量** (samples/sec): 越高越好
- **平均batch时间** (ms): 越低越好

典型性能参考（BERT-base, batch_size=16, seq_len=128, A100 GPU）:
- 训练: ~10-15 ms/batch
- 推理: ~5-8 ms/batch

## 优化目标

根据profiling结果，确定优化的算子优先级：

### 优先级1 (必做)
- [x] **Softmax**: 实现高效的warp-level reduction
- [x] **LayerNorm**: 使用Welford算法优化

### 优先级2 (推荐)
- [ ] **GEMM**: 实现高效的矩阵乘法（不使用cuBLAS）
- [ ] **GELU**: 优化激活函数计算

### 优先级3 (高级)
- [ ] **Fused Attention**: 融合QKV投影和attention计算
- [ ] **Fused FFN**: 融合Linear+GELU+Linear

## 常见问题

### Q1: 内存不足
**解决**: 减小batch_size
```bash
python train_bert_classification.py --batch_size 8 ...
```

### Q2: 数据集下载慢
**解决**: 使用镜像或预先下载
```python
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: Profiling时间太长
**解决**: 减少profiling的batch数量（修改脚本中的`num_batches`参数）

### Q4: 想测试更长的序列
**解决**: 增加max_length（注意显存）
```bash
python train_bert_classification.py --max_length 256 ...
```

## 下一步

1. ✅ 运行profiling，识别热点kernel
2. ⏭️ 分析kernel源码（在`aten/src/ATen/native/cuda/`目录）
3. ⏭️ 实现优化版本的算子
4. ⏭️ 对比优化前后的性能

## 相关文件

- `profile_bert.py`: 简化版的profiling脚本
- `train_bert_classification.py`: 完整的训练+profiling脚本
- `../02_算子调研/`: 算子源码分析文档
- `../04_implementation/`: 优化算子实现










