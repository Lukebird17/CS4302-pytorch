# 快速开始指南

这是一个 5 分钟快速上手指南。完整文档请参考 [README.md](README.md)。

---

## 1️⃣ 环境准备（2分钟）

```bash
# 确保 PyTorch 2.1.0 已安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
# 预期输出: PyTorch版本: 2.1.0

# 确保 CUDA 可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
# 预期输出: CUDA可用: True
```

如果版本不对，安装正确的 PyTorch：
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 2️⃣ 算子调研（1分钟）

测试 BERT 中哪些算子最耗时：

```bash
cd /path/to/lhl/operator_search

# 测试所有算子（约5分钟）
bash run_all_benchmarks.sh

# 或只测试 GEMM（最重要的算子）
python test_new.py --op addmm
```

**查看结果：**
```bash
cat output/addmm/imdb_addmm_final.csv
```

---

## 3️⃣ 融合算子安装（2分钟）

```bash
cd /path/to/lhl/bert_inference_acceleration

# 一键安装
bash install.sh
```

如果失败，手动安装：
```bash
cd custom_ops
rm -rf build dist *.so
pip install -e . --no-build-isolation
```

---

## 4️⃣ 验证正确性（30秒）

```bash
cd /path/to/lhl/bert_inference_acceleration

# 运行正确性测试
python tests/test_correctness.py
```

**预期输出：**
```
✅ 所有针对 BERT 场景的算子验证通过！
```

---

## 5️⃣ 性能测试（1分钟）

```bash
# 多数据集测试
python test_multi_dataset_performance.py

# IMDB 详细测试
python test_imdb_performance.py
```

---

## 🎯 核心命令速查

| 任务 | 命令 |
|------|------|
| 算子调研 - 全部 | `cd operator_search && bash run_all_benchmarks.sh` |
| 算子调研 - GEMM | `cd operator_search && python test_new.py --op addmm` |
| 安装融合算子 | `cd bert_inference_acceleration && bash install.sh` |
| 正确性测试 | `cd bert_inference_acceleration && python tests/test_correctness.py` |
| 性能测试 | `cd bert_inference_acceleration && python test_multi_dataset_performance.py` |

---

## ⚠️ 常见问题

**Q: 编译失败？**
```bash
# 检查 CUDA 是否正确安装
nvcc --version

# 设置 CUDA 路径
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Q: 导入失败？**
```bash
# 设置 PyTorch 库路径
export LD_LIBRARY_PATH=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH
```

**Q: 数据集在哪？**
- IMDB: `/hy-tmp/lhl/bert_inference_acceleration/dataset/imdb/`
- AG News: `/hy-tmp/lhl/bert_inference_acceleration/dataset/ag_news/`

---

## 📖 详细文档

- [完整 README](README.md) - 详细的技术文档
- [技术解释](bert_inference_acceleration/TECHNICAL_EXPLANATION.md) - 深入的技术原理

---

**有问题？** 查看完整 [README.md](README.md) 的"常见问题"章节。
