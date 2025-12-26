#!/bin/bash

# 快速修复pyarrow版本兼容性问题的脚本

echo "========================================"
echo "修复pyarrow版本兼容性问题"
echo "========================================"
echo ""

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到conda环境"
    echo "建议在conda环境中运行此脚本"
    echo ""
fi

echo "问题原因："
echo "  datasets库与当前安装的pyarrow版本不兼容"
echo "  错误信息: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'"
echo ""

echo "解决方案选项："
echo ""
echo "1. 升级pyarrow到最新版本（推荐）"
echo "2. 降级pyarrow到兼容版本"
echo "3. 重新安装datasets库"
echo "4. 不修复（跳过IMDB数据集，使用模拟数据）"
echo ""

read -p "请选择解决方案 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "正在升级pyarrow..."
        pip install --upgrade pyarrow
        if [ $? -eq 0 ]; then
            echo "✓ pyarrow升级成功"
            echo "正在验证..."
            python -c "import datasets; print('✓ datasets库可以正常导入')" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✓ 问题已解决!"
            else
                echo "⚠ datasets仍然无法导入，请尝试其他方案"
            fi
        else
            echo "✗ pyarrow升级失败"
        fi
        ;;
    2)
        echo ""
        echo "正在降级pyarrow到12.0.1..."
        pip install pyarrow==12.0.1
        if [ $? -eq 0 ]; then
            echo "✓ pyarrow降级成功"
            echo "正在验证..."
            python -c "import datasets; print('✓ datasets库可以正常导入')" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✓ 问题已解决!"
            else
                echo "⚠ datasets仍然无法导入，请尝试其他方案"
            fi
        else
            echo "✗ pyarrow降级失败"
        fi
        ;;
    3)
        echo ""
        echo "正在重新安装datasets..."
        pip uninstall -y datasets
        pip install datasets
        if [ $? -eq 0 ]; then
            echo "✓ datasets重新安装成功"
            echo "正在验证..."
            python -c "import datasets; print('✓ datasets库可以正常导入')" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✓ 问题已解决!"
            else
                echo "⚠ datasets仍然无法导入，请尝试方案1或2"
            fi
        else
            echo "✗ datasets重新安装失败"
        fi
        ;;
    4)
        echo ""
        echo "跳过修复"
        echo ""
        echo "注意事项："
        echo "  - BERT推理评测将使用模拟数据而非真实的IMDB数据集"
        echo "  - 这不影响LayerNorm的调研和性能测试"
        echo "  - 模拟数据同样可以用于性能评测"
        echo ""
        echo "可以继续进行实验:"
        echo "  python test_installation.py    # 重新检查环境"
        echo "  python quick_demo.py           # 快速演示"
        echo "  python layernorm_research.py   # LayerNorm调研"
        ;;
    *)
        echo ""
        echo "无效的选择"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "处理完成"
echo "========================================"
echo ""
echo "建议运行测试验证:"
echo "  python test_installation.py"
echo ""

