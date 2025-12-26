#!/bin/bash

# BERT 算子融合优化 - 完整评测脚本

echo "========================================================================"
echo "BERT 算子融合优化 - 完整评测流程"
echo "========================================================================"
echo ""

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到nvcc，请确保CUDA已安装"
    exit 1
fi

# 检查Python
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 步骤1: 编译融合算子
echo "步骤1: 编译融合算子"
echo "----------------------------------------"
cd custom_ops
python setup.py install
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi
echo "✓ 编译成功"
echo ""

# 步骤2: 测试融合算子
echo "步骤2: 测试融合算子"
echo "----------------------------------------"
python fused_ops_wrapper.py
if [ $? -ne 0 ]; then
    echo "❌ 测试失败"
    exit 1
fi
echo "✓ 测试通过"
echo ""

# 步骤3: 测试优化模型
echo "步骤3: 测试优化模型"
echo "----------------------------------------"
cd ../models
python bert_optimized.py
if [ $? -ne 0 ]; then
    echo "❌ 模型测试失败"
    exit 1
fi
echo "✓ 模型测试通过"
echo ""

# 步骤4: Baseline评测
echo "步骤4: 运行 Baseline 评测"
echo "----------------------------------------"
cd ../benchmark
python baseline_benchmark.py
if [ $? -ne 0 ]; then
    echo "❌ Baseline评测失败"
    exit 1
fi
echo "✓ Baseline评测完成"
echo ""

# 步骤5: 优化版评测
echo "步骤5: 运行优化版评测"
echo "----------------------------------------"
python optimized_benchmark.py
if [ $? -ne 0 ]; then
    echo "❌ 优化版评测失败"
    exit 1
fi
echo "✓ 优化版评测完成"
echo ""

# 步骤6: 对比结果
echo "步骤6: 对比结果"
echo "----------------------------------------"
python compare_results.py
if [ $? -ne 0 ]; then
    echo "❌ 结果对比失败"
    exit 1
fi
echo "✓ 结果对比完成"
echo ""

# 完成
echo "========================================================================"
echo "✓ 所有步骤完成！"
echo "========================================================================"
echo ""
echo "生成的文件:"
echo "  - ../results/baseline_results.json       # Baseline评测结果"
echo "  - ../results/optimized_results.json      # 优化版评测结果"
echo "  - ../results/comparison_report.md        # 对比报告"
echo ""
echo "查看报告:"
echo "  cat ../results/comparison_report.md"
echo ""


