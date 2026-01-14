#!/bin/bash

# BERT推理加速 - 完整测试步骤（Luke环境）

echo "=================================="
echo "BERT推理加速 - 测试流程"
echo "=================================="

cd /hy-tmp/lhl/bert_inference_acceleration

# 步骤1: 验证编译
echo ""
echo "步骤1: 验证编译是否成功"
echo "----------------------------------"
python3 << 'EOF'
import sys
try:
    import custom_ops
    ops = [x for x in dir(custom_ops) if not x.startswith('_') and x != 'torch' and callable(getattr(custom_ops, x, None))]
    if len(ops) >= 4:
        print("✅ 编译成功!")
        print(f"   找到 {len(ops)} 个算子:")
        for op in ops:
            print(f"   - {op}")
    else:
        print("⚠️  编译可能有问题，只找到", len(ops), "个算子")
        print("   尝试重新编译: cd custom_ops && pip install -e . --no-build-isolation")
        sys.exit(1)
except Exception as e:
    print("❌ 导入失败:", e)
    print("   请先编译: cd custom_ops && pip install -e . --no-build-isolation")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 验证失败，请重新编译"
    exit 1
fi

# 步骤2: 运行正确性测试
echo ""
echo "步骤2: 运行正确性测试"
echo "----------------------------------"
python tests/test_correctness.py 2>&1 | tail -30

# 步骤3: 运行性能对比测试
echo ""
echo "步骤3: 运行性能对比测试"
echo "----------------------------------"
python benchmarks/benchmark.py --num_iters 50

# 步骤4: 运行IMDB场景测试（如果有融合算子）
echo ""
echo "步骤4: 运行IMDB场景性能测试"
echo "----------------------------------"
python test_imdb_performance.py

echo ""
echo "=================================="
echo "✅ 所有测试完成！"
echo "=================================="




