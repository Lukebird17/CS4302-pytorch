#!/bin/bash
# 全面的CUDA算子Profiling运行脚本
# 用于收集Transformer/BERT模型的性能数据

set -e  # 遇到错误立即退出

echo "=========================================="
echo "🚀 BERT CUDA算子全面Profiling"
echo "=========================================="

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: 未检测到NVIDIA GPU或nvidia-smi命令"
    exit 1
fi

echo ""
echo "📊 GPU信息:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# 创建输出目录
OUTPUT_DIR="./profiling_results"
mkdir -p "$OUTPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"
echo ""

# 检查Python环境
echo "🐍 检查Python环境..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
if python3 -c "import transformers" 2>/dev/null; then
    echo "✅ transformers库已安装"
    USE_REAL_BERT="--use-real-bert"
else
    echo "⚠️  transformers库未安装，将使用简化模型"
    USE_REAL_BERT=""
fi
echo ""

# 运行profiling
echo "=========================================="
echo "⏳ 开始Profiling..."
echo "=========================================="
echo ""

# 方案1: 快速测试（推荐首次运行）
echo "方案1: 快速profiling（小规模测试）"
python3 profile_bert.py \
    $USE_REAL_BERT \
    --batch-sizes 1 8 \
    --seq-lens 128 \
    --hidden-size 768 \
    --output-dir "$OUTPUT_DIR/quick_test" \
    2>&1 | tee "$OUTPUT_DIR/quick_test.log"

echo ""
echo "✅ 快速测试完成！"
echo ""

# 方案2: 完整测试（可选）
read -p "是否运行完整测试？(包含更多配置，耗时较长) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "方案2: 完整profiling"
    python3 profile_bert.py \
        $USE_REAL_BERT \
        --batch-sizes 1 4 8 16 \
        --seq-lens 128 256 512 \
        --hidden-size 768 \
        --output-dir "$OUTPUT_DIR/full_test" \
        2>&1 | tee "$OUTPUT_DIR/full_test.log"
    
    echo ""
    echo "✅ 完整测试完成！"
fi

echo ""
echo "=========================================="
echo "📊 Profiling结果总结"
echo "=========================================="
echo ""

# 列出生成的文件
echo "生成的文件:"
find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.md" | sort

echo ""
echo "=========================================="
echo "✅ Profiling全部完成！"
echo "=========================================="
echo ""
echo "📝 下一步操作:"
echo "  1. 查看Markdown报告: ls -lh $OUTPUT_DIR/*/*.md"
echo "  2. 查看JSON统计: cat $OUTPUT_DIR/*/profiling_stats_*.json | jq '.top_aten_operators[0:3]'"
echo "  3. Chrome可视化: 在浏览器打开 chrome://tracing，加载 *_trace.json 文件"
echo "  4. 开始算子源码调研"
echo ""
echo "📂 结果目录: $(cd $OUTPUT_DIR && pwd)"








