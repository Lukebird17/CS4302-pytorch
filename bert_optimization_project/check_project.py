#!/usr/bin/env python3
"""
项目完整性检查脚本
检查所有CUDA kernel、Python接口、模型集成是否正确
"""

import sys
import torch

print("="*80)
print("BERT 算子优化项目 - 完整性检查")
print("="*80)

errors = []
warnings = []

# ============================================================================
# 检查1: CUDA算子加载
# ============================================================================
print("\n【检查1】CUDA算子加载")
print("-"*80)

try:
    import bert_fused_ops
    print("✓ bert_fused_ops 模块加载成功")
    
    # 检查导出的函数
    required_fused_ops = [
        'fused_ln_residual_optimized',
        'fast_gelu',
        'optimized_softmax',
        'bias_gelu_fusion'
    ]
    
    for func_name in required_fused_ops:
        if hasattr(bert_fused_ops, func_name):
            print(f"  ✓ {func_name}")
        else:
            errors.append(f"bert_fused_ops 缺少函数: {func_name}")
            print(f"  ✗ {func_name} - 缺失！")
            
except ImportError as e:
    errors.append(f"无法加载 bert_fused_ops: {e}")
    print(f"✗ bert_fused_ops 加载失败: {e}")

try:
    import bert_custom_gemm
    print("\n✓ bert_custom_gemm 模块加载成功")
    
    # 检查导出的函数
    required_gemm_ops = [
        'custom_gemm',
        'custom_gemm_bias_gelu'
    ]
    
    for func_name in required_gemm_ops:
        if hasattr(bert_custom_gemm, func_name):
            print(f"  ✓ {func_name}")
        else:
            errors.append(f"bert_custom_gemm 缺少函数: {func_name}")
            print(f"  ✗ {func_name} - 缺失！")
            
except ImportError as e:
    errors.append(f"无法加载 bert_custom_gemm: {e}")
    print(f"✗ bert_custom_gemm 加载失败: {e}")


# ============================================================================
# 检查2: CUDA kernel功能测试
# ============================================================================
print("\n【检查2】CUDA kernel功能测试")
print("-"*80)

if torch.cuda.is_available():
    print("✓ CUDA可用")
    
    # 测试 fused_ln_residual_optimized
    try:
        x = torch.randn(2, 768).cuda()
        residual = torch.randn(2, 768).cuda()
        gamma = torch.ones(768).cuda()
        beta = torch.zeros(768).cuda()
        
        output = bert_fused_ops.fused_ln_residual_optimized(
            x, residual, gamma, beta, 1e-5
        )
        assert output.shape == (2, 768), "输出形状错误"
        print("  ✓ fused_ln_residual_optimized 功能正常")
    except Exception as e:
        errors.append(f"fused_ln_residual_optimized 测试失败: {e}")
        print(f"  ✗ fused_ln_residual_optimized: {e}")
    
    # 测试 fast_gelu
    try:
        x = torch.randn(2, 768).cuda()
        output = bert_fused_ops.fast_gelu(x)
        assert output.shape == x.shape, "输出形状错误"
        print("  ✓ fast_gelu 功能正常")
    except Exception as e:
        errors.append(f"fast_gelu 测试失败: {e}")
        print(f"  ✗ fast_gelu: {e}")
    
    # 测试 custom_gemm
    try:
        A = torch.randn(128, 768).cuda()
        B = torch.randn(768, 768).cuda()
        C = bert_custom_gemm.custom_gemm(A, B)
        assert C.shape == (128, 768), "输出形状错误"
        
        # 验证正确性（与PyTorch对比）
        C_torch = torch.mm(A, B)
        diff = torch.abs(C - C_torch).max().item()
        if diff < 1e-3:
            print(f"  ✓ custom_gemm 功能正常 (误差={diff:.2e})")
        else:
            warnings.append(f"custom_gemm 精度较低: 最大误差={diff:.2e}")
            print(f"  ⚠ custom_gemm 精度较低: 最大误差={diff:.2e}")
    except Exception as e:
        errors.append(f"custom_gemm 测试失败: {e}")
        print(f"  ✗ custom_gemm: {e}")
    
    # 测试 custom_gemm_bias_gelu
    try:
        A = torch.randn(128, 768).cuda()
        B = torch.randn(768, 3072).cuda()
        bias = torch.randn(3072).cuda()
        C = bert_custom_gemm.custom_gemm_bias_gelu(A, B, bias)
        assert C.shape == (128, 3072), "输出形状错误"
        print("  ✓ custom_gemm_bias_gelu 功能正常")
    except Exception as e:
        errors.append(f"custom_gemm_bias_gelu 测试失败: {e}")
        print(f"  ✗ custom_gemm_bias_gelu: {e}")
        
else:
    warnings.append("CUDA不可用，跳过kernel测试")
    print("⚠ CUDA不可用，跳过kernel测试")


# ============================================================================
# 检查3: 模型集成
# ============================================================================
print("\n【检查3】模型集成检查")
print("-"*80)

try:
    sys.path.append('models')
    from bert_optimized import create_optimized_bert, CustomLinear
    print("✓ bert_optimized 模块导入成功")
    
    # 检查CustomLinear
    try:
        linear = CustomLinear(768, 768)
        print("  ✓ CustomLinear 类定义正确")
    except Exception as e:
        errors.append(f"CustomLinear 创建失败: {e}")
        print(f"  ✗ CustomLinear: {e}")
    
    # 检查模型创建
    if torch.cuda.is_available():
        try:
            model = create_optimized_bert("bert-base-uncased")
            print("  ✓ create_optimized_bert 函数正常")
            
            # 检查是否替换了Linear
            linear_count = 0
            custom_linear_count = 0
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_count += 1
                elif isinstance(module, CustomLinear):
                    custom_linear_count += 1
            
            if custom_linear_count > 0:
                print(f"  ✓ 成功替换 {custom_linear_count} 个Linear层为CustomLinear")
            else:
                warnings.append("未替换任何Linear层")
                print(f"  ⚠ 未替换任何Linear层")
            
            if linear_count > 0:
                warnings.append(f"还有 {linear_count} 个Linear层未替换")
                print(f"  ⚠ 还有 {linear_count} 个Linear层未替换")
                
        except Exception as e:
            errors.append(f"create_optimized_bert 失败: {e}")
            print(f"  ✗ create_optimized_bert: {e}")
    
except ImportError as e:
    errors.append(f"无法导入 bert_optimized: {e}")
    print(f"✗ bert_optimized 导入失败: {e}")


# ============================================================================
# 检查4: 文件完整性
# ============================================================================
print("\n【检查4】文件完整性")
print("-"*80)

import os

required_files = {
    'custom_ops/fused_ops.cu': 'CUDA融合算子源码',
    'custom_ops/custom_gemm.cu': '自定义GEMM源码',
    'custom_ops/setup.py': '编译脚本',
    'models/bert_optimized.py': '优化模型',
    'test_performance.py': '性能测试脚本',
    'README.md': '项目说明',
}

for file_path, description in required_files.items():
    if os.path.exists(file_path):
        print(f"  ✓ {file_path} - {description}")
    else:
        errors.append(f"缺少文件: {file_path}")
        print(f"  ✗ {file_path} - 缺失！")


# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("检查总结")
print("="*80)

if errors:
    print(f"\n❌ 发现 {len(errors)} 个错误：")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")

if warnings:
    print(f"\n⚠️  发现 {len(warnings)} 个警告：")
    for i, warn in enumerate(warnings, 1):
        print(f"  {i}. {warn}")

if not errors and not warnings:
    print("\n✅ 所有检查通过！项目完整性良好。")
elif not errors:
    print(f"\n✅ 主要功能正常，但有 {len(warnings)} 个警告需要注意。")
else:
    print(f"\n❌ 项目存在问题，需要修复 {len(errors)} 个错误。")
    sys.exit(1)

print("\n" + "="*80)


