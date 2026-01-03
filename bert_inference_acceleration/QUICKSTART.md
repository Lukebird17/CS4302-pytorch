# 🚀 快速开始指南

## 一键安装和测试

### 在Luke环境中运行

```bash
# 1. 激活环境
conda activate luke

# 2. 进入项目目录
cd /hy-tmp/lhl/bert_inference_acceleration

# 3. 安装（清理旧版本+编译新版本）
./reinstall.sh

# 4. 运行IMDB性能测试
python test_imdb_performance.py
```

## 预期结果

### 正确性测试
```bash
python tests/test_correctness.py
```
**预期输出**：
- GEMM: ✅ 通过
- LayerNorm: ✅ 通过
- GEMM+Bias: ✅ 通过
- GEMM+Bias+GELU: ✅ 通过
- GEMM+Bias+Add+LayerNorm: ✅ 通过
- GEMM+Bias+GELU+Add+LayerNorm: ✅ 通过

### IMDB性能测试
```bash
python test_imdb_performance.py
```
**预期输出**：
- Attention输出层：融合算子达到PyTorch **88%** 性能
- FFN第二层：融合算子达到PyTorch **88%** 性能
- Kernel数量：5-6个 → 1个

### 基准测试
```bash
python benchmarks/benchmark.py
```
**预期输出**：
- GEMM: 达到cuBLAS **73%** 性能
- 所有算子误差 < 1e-4

## 故障排除

### 问题1: 性能异常慢（0.08x而不是0.88x）

**原因**：使用了旧版本的.so文件

**解决**：
```bash
./reinstall.sh  # 强制重新编译
```

### 问题2: 导入失败

**原因**：PyTorch库路径未设置

**解决**：
```bash
export LD_LIBRARY_PATH=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH
```

### 问题3: HF下载慢

**原因**：未配置HF Mirror

**解决**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 核心文件说明

| 文件/目录 | 说明 |
|---------|------|
| `custom_ops/custom_gemm.cu` | **核心**：GEMM及融合算子CUDA实现 |
| `tests/test_correctness.py` | 正确性验证 |
| `test_imdb_performance.py` | IMDB场景性能对比 |
| `benchmarks/benchmark.py` | 基准性能测试 |
| `reinstall.sh` | 重新安装脚本 |
| `TECHNICAL_EXPLANATION.md` | 技术详解 |
| `FINAL_SUMMARY.md` | 最终成果 |

## 性能指标总结

| 指标 | 目标 | 实际 | 状态 |
|-----|------|------|------|
| GEMM vs cuBLAS | 80% | 73% | ⚠️ 接近 |
| 融合算子 vs PyTorch | - | 88% | ✅ 优秀 |
| 正确性 | < 1e-4 | < 1e-5 | ✅ 优秀 |
| Kernel融合 | 5-6→1 | 5-6→1 | ✅ 达成 |

## 下一步优化方向

如果要进一步提升GEMM性能到80%+：
1. 调整Tile大小（当前128x128x8）
2. 尝试Warp-level GEMM
3. 使用Tensor Core（半精度）
4. 针对特定矩阵尺寸优化

详见 `TECHNICAL_EXPLANATION.md`

