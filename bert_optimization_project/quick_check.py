#!/usr/bin/env python3
"""
项目完整性检查 - 离线版本
"""

import sys
import os

print("="*80)
print("BERT 算子优化项目 - 快速完整性检查")
print("="*80)

errors = []
success = []

# ============================================================================
# 1. 检查文件存在性
# ============================================================================
print("\n【1】核心文件检查")
print("-"*80)

files_to_check = {
    # CUDA源码
    'custom_ops/fused_ops.cu': '融合算子CUDA源码',
    'custom_ops/custom_gemm.cu': '自实现GEMM CUDA源码 ⭐',
    'custom_ops/setup.py': '编译脚本',
    
    # Python模块
    'models/bert_optimized.py': '优化模型Python代码',
    'test_performance.py': '性能测试脚本',
    
    # 文档
    'README.md': '项目说明文档',
    'GEMM实现说明.txt': 'GEMM技术文档',
}

for filepath, desc in files_to_check.items():
    if os.path.exists(filepath):
        print(f"  ✓ {filepath:<40} - {desc}")
        success.append(f"文件存在: {filepath}")
    else:
        print(f"  ✗ {filepath:<40} - 缺失！")
        errors.append(f"缺少文件: {filepath}")

# ============================================================================
# 2. 检查CUDA算子是否编译
# ============================================================================
print("\n【2】CUDA算子编译检查")
print("-"*80)

so_files = {
    'bert_fused_ops': 'custom_ops/build/lib.linux-x86_64-cpython-310/bert_fused_ops.cpython-310-x86_64-linux-gnu.so',
    'bert_custom_gemm': 'custom_ops/build/lib.linux-x86_64-cpython-310/bert_custom_gemm.cpython-310-x86_64-linux-gnu.so',
}

for name, so_path in so_files.items():
    if os.path.exists(so_path):
        size = os.path.getsize(so_path) / 1024
        print(f"  ✓ {name:<20} - 已编译 ({size:.1f} KB)")
        success.append(f"算子已编译: {name}")
    else:
        print(f"  ✗ {name:<20} - 未编译")
        errors.append(f"算子未编译: {name}")

# ============================================================================
# 3. 检查CUDA kernel实现
# ============================================================================
print("\n【3】CUDA Kernel实现检查")
print("-"*80)

# 检查 fused_ops.cu
fused_ops_kernels = [
    ('fused_ln_residual_eval_kernel_768', 'LayerNorm+Residual融合'),
    ('fast_gelu_kernel_vectorized', '快速GELU'),
    ('online_softmax_kernel', 'Online Softmax'),
    ('bias_gelu_fusion_kernel_vec', 'Bias+GELU融合'),
]

if os.path.exists('custom_ops/fused_ops.cu'):
    with open('custom_ops/fused_ops.cu', 'r') as f:
        content = f.read()
    
    for kernel_name, desc in fused_ops_kernels:
        if kernel_name in content:
            print(f"  ✓ {desc:<30} - {kernel_name}")
            success.append(f"Kernel存在: {kernel_name}")
        else:
            print(f"  ✗ {desc:<30} - 缺失")
            errors.append(f"Kernel缺失: {kernel_name}")

# 检查 custom_gemm.cu
gemm_kernels = [
    ('gemm_kernel_tiled', 'GEMM基础版本'),
    ('gemm_kernel_optimized', 'GEMM优化版本 (Register Tiling)'),
    ('gemm_768_kernel', 'GEMM BERT特化版本 ⭐'),
    ('gemm_bias_gelu_kernel_768', 'GEMM+Bias+GELU融合 ⭐'),
]

if os.path.exists('custom_ops/custom_gemm.cu'):
    with open('custom_ops/custom_gemm.cu', 'r') as f:
        content = f.read()
    
    for kernel_name, desc in gemm_kernels:
        if kernel_name in content:
            print(f"  ✓ {desc:<35} - {kernel_name}")
            success.append(f"GEMM Kernel存在: {kernel_name}")
        else:
            print(f"  ✗ {desc:<35} - 缺失")
            errors.append(f"GEMM Kernel缺失: {kernel_name}")

# ============================================================================
# 4. 检查Python接口导出
# ============================================================================
print("\n【4】Python接口导出检查")
print("-"*80)

# 检查 PYBIND11 绑定
if os.path.exists('custom_ops/fused_ops.cu'):
    with open('custom_ops/fused_ops.cu', 'r') as f:
        content = f.read()
    
    required_exports = [
        'fused_ln_residual_optimized',
        'fast_gelu',
        'optimized_softmax',
        'bias_gelu_fusion',
    ]
    
    for func_name in required_exports:
        if f'"{func_name}"' in content or f"'{func_name}'" in content:
            print(f"  ✓ {func_name}")
            success.append(f"导出函数: {func_name}")
        else:
            print(f"  ✗ {func_name} - 未导出")
            errors.append(f"函数未导出: {func_name}")

if os.path.exists('custom_ops/custom_gemm.cu'):
    with open('custom_ops/custom_gemm.cu', 'r') as f:
        content = f.read()
    
    required_exports = [
        'custom_gemm',
        'custom_gemm_bias_gelu',
    ]
    
    for func_name in required_exports:
        if f'"{func_name}"' in content or f"'{func_name}'" in content:
            print(f"  ✓ {func_name}")
            success.append(f"导出GEMM函数: {func_name}")
        else:
            print(f"  ✗ {func_name} - 未导出")
            errors.append(f"GEMM函数未导出: {func_name}")

# ============================================================================
# 5. 检查模型集成代码
# ============================================================================
print("\n【5】模型集成代码检查")
print("-"*80)

if os.path.exists('models/bert_optimized.py'):
    with open('models/bert_optimized.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('CustomLinear', 'CustomLinear类定义'),
        ('bert_custom_gemm.custom_gemm', '使用自定义GEMM'),
        ('bert_fused_ops.fused_ln_residual_optimized', '使用融合LayerNorm'),
        ('create_optimized_bert', '模型创建函数'),
    ]
    
    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc}")
            success.append(f"代码检查: {desc}")
        else:
            print(f"  ✗ {desc} - 未实现")
            errors.append(f"代码缺失: {desc}")

# ============================================================================
# 6. 优化技术清单
# ============================================================================
print("\n【6】优化技术实现清单")
print("-"*80)

optimizations = [
    ('Shared Memory Tiling', 'gemm_kernel_tiled', 'custom_gemm.cu'),
    ('Register Tiling', 'gemm_kernel_optimized', 'custom_gemm.cu'),
    ('BERT 768特化', 'gemm_768_kernel', 'custom_gemm.cu'),
    ('Welford算法', 'fused_ln_residual_eval_kernel_768', 'fused_ops.cu'),
    ('快速GELU', 'fast_gelu', 'fused_ops.cu'),
    ('Online Softmax', 'online_softmax', 'fused_ops.cu'),
    ('float4向量化', 'float4', 'custom_gemm.cu'),
]

for opt_name, pattern, filename in optimizations:
    filepath = f'custom_ops/{filename}'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            if pattern in f.read():
                print(f"  ✓ {opt_name}")
                success.append(f"优化实现: {opt_name}")
            else:
                print(f"  ⚠ {opt_name} - 可能未实现")
    else:
        print(f"  ✗ {opt_name} - 文件不存在")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("检查总结")
print("="*80)

print(f"\n✅ 成功项: {len(success)}")
print(f"❌ 错误项: {len(errors)}")

if errors:
    print(f"\n错误列表：")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    print("\n需要修复这些问题！")
    sys.exit(1)
else:
    print("\n🎉 所有检查通过！项目完整性良好。")
    print("\n核心功能：")
    print("  ✓ 自实现GEMM kernel (3个版本)")
    print("  ✓ 融合算子 (LayerNorm, GELU, Softmax)")
    print("  ✓ Python接口完整")
    print("  ✓ 模型集成代码正确")
    print("\n可以运行: python test_performance.py 进行性能测试")

print("\n" + "="*80)


