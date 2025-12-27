#!/bin/bash
# 一键完成所有测试并保存截图用的输出

set -e  # 遇到错误立即退出

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "BERT 算子优化项目 - 一键测试脚本"
echo "================================================================================"
echo ""
echo "本脚本将依次完成："
echo "  1. 检查环境"
echo "  2. 编译CUDA算子"
echo "  3. 运行功能验证"
echo "  4. 运行性能测试"
echo "  5. 保存所有输出用于截图"
echo ""
echo "预计总时间: 10-15分钟"
echo "================================================================================"
echo ""

# ============================================================================
# 步骤1: 环境检查
# ============================================================================
echo "【步骤1/5】环境检查"
echo "--------------------------------------------------------------------------------"

# 检查CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    echo "✓ CUDA: $CUDA_VERSION"
else
    echo "❌ 错误: 未找到nvcc，请确保CUDA已安装"
    exit 1
fi

# 检查Python和PyTorch
python -c "import torch" 2>/dev/null || { echo "❌ 错误: PyTorch未安装"; exit 1; }
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "✓ PyTorch: $PYTORCH_VERSION"

CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✓ GPU: $GPU_NAME"
else
    echo "❌ 错误: CUDA不可用"
    exit 1
fi

echo ""
echo "✅ 环境检查通过！"
echo ""

# ============================================================================
# 步骤2: 编译CUDA算子
# ============================================================================
echo "【步骤2/5】编译CUDA算子"
echo "--------------------------------------------------------------------------------"

cd custom_ops

# 清理旧编译
echo "清理旧的编译文件..."
rm -rf build dist *.egg-info

# 编译
echo "开始编译CUDA算子（预计2-5分钟）..."
python setup.py install > ../compile_output.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "编译信息已保存到: compile_output.txt"
    echo "可用于截图的关键信息："
    grep -E "(ptxas info|Installed)" ../compile_output.txt | head -10
else
    echo "❌ 编译失败！请查看 compile_output.txt"
    exit 1
fi

cd ..
echo ""

# ============================================================================
# 步骤3: 功能验证
# ============================================================================
echo "【步骤3/5】功能验证"
echo "--------------------------------------------------------------------------------"

echo "运行完整性检查..."
python check_project.py > check_output.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 功能验证通过！"
    echo ""
    echo "验证结果已保存到: check_output.txt"
    echo "关键验证结果："
    tail -20 check_output.txt
else
    echo "⚠ 功能验证有警告，请查看 check_output.txt"
fi

echo ""

# ============================================================================
# 步骤4: 性能测试
# ============================================================================
echo "【步骤4/5】性能测试"
echo "--------------------------------------------------------------------------------"

echo "运行性能测试（预计5-10分钟）..."
echo "测试配置："
echo "  - Batch sizes: 1, 4, 8, 16, 32, 64"
echo "  - 序列长度: 128"
echo "  - 测试轮数: 5"
echo "  - 每轮迭代: 50"
echo ""

python test_performance.py > performance_output.txt 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 性能测试完成！"
    echo ""
    echo "测试结果已保存到: performance_output.txt"
    echo "性能对比结果："
    grep -A 20 "测试结果汇总" performance_output.txt
else
    echo "❌ 性能测试失败！请查看 performance_output.txt"
    exit 1
fi

echo ""

# ============================================================================
# 步骤5: 生成总结
# ============================================================================
echo "【步骤5/5】生成测试总结"
echo "--------------------------------------------------------------------------------"

cat > test_summary.txt << 'EOF'
===============================================================================
BERT 算子优化项目 - 测试总结报告
===============================================================================

本次测试完成时间: $(date '+%Y-%m-%d %H:%M:%S')

【1】环境信息
--------------------------------------------------------------------------------
CUDA版本: $(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
PyTorch版本: $(python -c "import torch; print(torch.__version__)")
GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "N/A")

【2】编译结果
--------------------------------------------------------------------------------
EOF

if grep -q "Installed" compile_output.txt; then
    echo "✅ CUDA算子编译成功" >> test_summary.txt
    echo "" >> test_summary.txt
    echo "编译信息:" >> test_summary.txt
    grep "ptxas info" compile_output.txt | tail -5 >> test_summary.txt
else
    echo "❌ 编译失败" >> test_summary.txt
fi

cat >> test_summary.txt << 'EOF'

【3】功能验证
--------------------------------------------------------------------------------
EOF

if grep -q "所有检查通过" check_output.txt; then
    echo "✅ 所有功能验证通过" >> test_summary.txt
    echo "" >> test_summary.txt
    grep -E "(✓|GEMM精度|Custom GEMM)" check_output.txt | head -10 >> test_summary.txt
else
    echo "⚠ 部分验证有警告" >> test_summary.txt
fi

cat >> test_summary.txt << 'EOF'

【4】性能测试结果
--------------------------------------------------------------------------------
EOF

if grep -q "测试结果汇总" performance_output.txt; then
    grep -A 20 "测试结果汇总" performance_output.txt >> test_summary.txt
else
    echo "❌ 性能测试失败" >> test_summary.txt
fi

echo ""
echo "✅ 测试总结已生成: test_summary.txt"
echo ""

# ============================================================================
# 完成
# ============================================================================
echo "================================================================================"
echo "🎉 所有测试完成！"
echo "================================================================================"
echo ""
echo "生成的文件（用于截图和报告）："
echo "  1. compile_output.txt     - 编译过程输出"
echo "  2. check_output.txt        - 功能验证输出"
echo "  3. performance_output.txt  - 性能测试输出"
echo "  4. test_summary.txt        - 测试总结"
echo ""
echo "下一步："
echo "  1. 查看 performance_output.txt，获取性能数据"
echo "  2. 使用 'less -R performance_output.txt' 查看完整输出"
echo "  3. 截图保存关键输出"
echo "  4. 填写实验报告"
echo ""
echo "推荐的截图命令："
echo "  tail -50 compile_output.txt    # 编译过程"
echo "  tail -30 check_output.txt      # 功能验证"
echo "  grep -A 20 '测试结果汇总' performance_output.txt  # 性能结果"
echo ""
echo "================================================================================"


