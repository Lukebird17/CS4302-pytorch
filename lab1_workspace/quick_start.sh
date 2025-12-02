#!/bin/bash

# CS4302 PyTorch CUDA算子优化 - 快速启动脚本

echo "========================================="
echo "CS4302 PyTorch CUDA算子优化大作业"
echo "========================================="

# 检查Python环境
echo -e "\n[1/5] 检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

python --version
echo "✓ Python环境正常"

# 检查CUDA
echo -e "\n[2/5] 检查CUDA环境..."
if ! command -v nvcc &> /dev/null; then
    echo "警告: 未找到nvcc"
else
    nvcc --version | head -4
    echo "✓ CUDA环境正常"
fi

# 检查依赖
echo -e "\n[3/5] 检查依赖包..."
pip list | grep -E "torch|transformers|datasets" || {
    echo "安装依赖包..."
    pip install torch transformers datasets tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
}
echo "✓ 依赖包已安装"

# 检查GPU
echo -e "\n[4/5] 检查GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU可用: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA版本: {torch.version.cuda}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('警告: GPU不可用')
"

# 运行快速测试
echo -e "\n[5/5] 运行快速测试..."
echo "这将训练一个小规模的BERT模型用于测试..."

cd 03_profiling

python train_bert_classification.py \
    --dataset imdb \
    --batch_size 8 \
    --epochs 1 \
    --do_train \
    --do_eval \
    --do_profile \
    --output_dir ../outputs/quick_test

echo -e "\n========================================="
echo "快速测试完成！"
echo "========================================="
echo ""
echo "输出文件位置:"
echo "  - 模型: outputs/quick_test/best_model/"
echo "  - Profiling: outputs/quick_test/profiling_results/"
echo ""
echo "下一步操作:"
echo "  1. 查看Chrome trace:"
echo "     打开 chrome://tracing"
echo "     加载 outputs/quick_test/profiling_results/bert_inference_trace.json"
echo ""
echo "  2. 查看kernel统计:"
echo "     cat outputs/quick_test/profiling_results/kernel_statistics.json"
echo ""
echo "  3. 开始算子调研:"
echo "     参考 02_算子调研/ 目录下的文档"
echo ""
echo "  4. 实现优化算子:"
echo "     在 04_implementation/src/ 目录下编写优化代码"
echo ""
echo "========================================="



