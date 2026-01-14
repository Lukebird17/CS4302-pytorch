#!/bin/bash
echo "========================================"
echo "BERT推理加速 - Luke环境安装"
echo "========================================"

# 检查是否在luke环境
if [[ "$CONDA_DEFAULT_ENV" != "luke" ]]; then
    echo "⚠️  请先激活luke环境："
    echo "   conda activate luke"
    exit 1
fi

echo ""
echo "步骤1: 配置HF Mirror..."
export HF_ENDPOINT=https://hf-mirror.com
echo "✓ HF_ENDPOINT=$HF_ENDPOINT"

echo ""
echo "步骤2: 清理旧版本..."
cd /hy-tmp/lhl/bert_inference_acceleration/custom_ops
rm -rf build dist *.egg-info *.so
pip uninstall -y custom_ops 2>/dev/null
echo "✓ 清理完成"

echo ""
echo "步骤3: 编译CUDA算子..."
pip install -e . --no-build-isolation

if [ $? -ne 0 ]; then
    echo "✗ 编译失败"
    exit 1
fi
echo "✓ 编译完成"

echo ""
echo "步骤4: 验证..."
cd /hy-tmp/lhl/bert_inference_acceleration
export LD_LIBRARY_PATH=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH

python << 'EOF'
from custom_ops_cuda import gemm
import torch

A = torch.randn(128, 128).cuda()
B = torch.randn(128, 128).cuda()
C = gemm(A, B, 1.0, 0.0)
print("✓ GEMM测试通过")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 安装成功！"
    echo "========================================"
    echo ""
    echo "运行测试:"
    echo "  python tests/test_correctness.py"
    echo "  python test_imdb_performance.py"
    echo "  python benchmarks/benchmark.py"
else
    echo "✗ 验证失败"
    exit 1
fi
