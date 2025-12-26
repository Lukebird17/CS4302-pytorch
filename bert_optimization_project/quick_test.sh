#!/bin/bash
set -e

# 激活环境
source /usr/local/miniconda3/bin/activate pytorch_test

echo "========================================================================"
echo "BERT 算子优化 - 性能测试（包含FP16 GEMM优化）"
echo "========================================================================"
echo ""
echo "测试3个版本:"
echo "  1. Baseline: 原生PyTorch"
echo "  2. Fused: CUDA融合算子"
echo "  3. FP16: CUDA融合算子 + FP16 Tensor Core（优化GEMM瓶颈）"
echo ""

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ 错误: CUDA未找到"
    exit 1
fi

echo "✓ CUDA版本: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo "✓ Python环境: $(which python)"
echo ""

# 编译算子（如果需要）
echo "========================================================================"
echo "步骤1: 检查并编译CUDA算子"
echo "========================================================================"

if python -c "import bert_fused_ops" 2>/dev/null; then
    echo "✓ CUDA算子已编译"
else
    echo "正在编译CUDA算子..."
    cd custom_ops
    rm -rf build dist *.egg-info
    python setup.py install > /dev/null 2>&1
    cd ..
    if python -c "import bert_fused_ops" 2>/dev/null; then
        echo "✓ CUDA算子编译成功"
    else
        echo "❌ CUDA算子编译失败"
        exit 1
    fi
fi

echo ""
echo "========================================================================"
echo "步骤2: 运行性能测试"
echo "========================================================================"
echo ""

python test_performance.py

echo ""
echo "========================================================================"
echo "测试完成！"
echo "========================================================================"

