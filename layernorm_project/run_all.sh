#!/bin/bash

# LayerNorm项目完整运行脚本
# 按照推荐顺序执行所有实验步骤

echo "========================================"
echo "LayerNorm CUDA算子调研与优化项目"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python${NC}"
    exit 1
fi

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}警告: 未找到nvcc，可能无法编译CUDA扩展${NC}"
fi

echo -e "${GREEN}步骤1: 环境检查${NC}"
echo "----------------------------------------"
python test_installation.py
if [ $? -ne 0 ]; then
    echo -e "${RED}环境检查失败，请先解决环境问题${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}步骤2: 快速演示${NC}"
echo "----------------------------------------"
python quick_demo.py

echo ""
echo -e "${GREEN}步骤3: LayerNorm深度调研${NC}"
echo "----------------------------------------"
python layernorm_research.py

echo ""
echo -e "${GREEN}步骤4: BERT推理评测${NC}"
echo "----------------------------------------"
python bert_inference_benchmark.py

echo ""
echo -e "${GREEN}步骤5: 编译自定义CUDA扩展${NC}"
echo "----------------------------------------"
read -p "是否编译自定义CUDA扩展? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python setup.py install
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}编译成功!${NC}"
        
        echo ""
        echo -e "${GREEN}步骤6: 测试自定义实现${NC}"
        echo "----------------------------------------"
        python custom_layernorm.py
        
        echo ""
        echo -e "${GREEN}步骤7: 性能对比测试${NC}"
        echo "----------------------------------------"
        python performance_comparison.py
    else
        echo -e "${RED}编译失败，跳过性能对比${NC}"
    fi
else
    echo -e "${YELLOW}跳过编译和性能对比${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}所有步骤完成!${NC}"
echo "========================================"
echo ""
echo "生成的文件:"
echo "  - profiler_results/          # Profiler结果"
echo "  - benchmark_results.json     # BERT评测结果"
echo "  - performance_comparison_results.json  # 性能对比"
echo "  - performance_plots/         # 性能图表"
echo "  - layernorm_research_report.txt  # 调研报告"
echo ""
echo "下一步:"
echo "  1. 查看调研报告: cat layernorm_research_report.txt"
echo "  2. 查看性能对比: cat performance_comparison_results.json"
echo "  3. 在Chrome中查看trace: chrome://tracing"
echo "  4. 撰写实验报告"
echo ""

