"""
正确性验证脚本
"""
import torch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import custom_ops
    USE_CUSTOM_OPS = True
    print("✓ 成功加载自定义算子")
    print(f"  可用函数: {[x for x in dir(custom_ops) if not x.startswith('_')]}")
except ImportError as e:
    USE_CUSTOM_OPS = False
    print(f"错误: 自定义算子未正确编译 - {e}")
    print("请运行: cd custom_ops && pip install -e .")
    sys.exit(1)


def test_gemm():
    """测试GEMM正确性"""
    print("\n" + "=" * 60)
    print("测试GEMM正确性")
    print("=" * 60)
    
    test_cases = [
        (10, 20, 30),
        (128, 768, 768),
        (512, 3072, 768),
        (1, 768, 768),      # 边界情况
        (1024, 1024, 1024), # 大矩阵
    ]
    
    all_passed = True
    
    for M, N, K in test_cases:
        A = torch.randn(M, K, device='cuda')
        B = torch.randn(K, N, device='cuda')
        
        # PyTorch实现
        C_pytorch = torch.matmul(A, B)
        
        # 自定义实现
        C_custom = custom_ops.gemm(A, B, 1.0, 0.0)
        
        # 计算误差
        max_diff = torch.max(torch.abs(C_custom - C_pytorch)).item()
        mean_diff = torch.mean(torch.abs(C_custom - C_pytorch)).item()
        relative_error = mean_diff / (torch.mean(torch.abs(C_pytorch)).item() + 1e-10)
        
        # 判断是否通过 (相对误差小于1e-5)
        passed = relative_error < 1e-5
        status = "✓ 通过" if passed else "✗ 失败"
        
        print(f"  [{M}x{K}] @ [{K}x{N}]: {status}")
        print(f"    最大误差: {max_diff:.2e}, 平均误差: {mean_diff:.2e}, 相对误差: {relative_error:.2e}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_gemm_bias():
    """测试GEMM+Bias正确性"""
    print("\n" + "=" * 60)
    print("测试GEMM+Bias正确性")
    print("=" * 60)
    
    test_cases = [
        (10, 20, 30),
        (128, 768, 768),
        (512, 3072, 768),
    ]
    
    all_passed = True
    
    for M, N, K in test_cases:
        A = torch.randn(M, K, device='cuda')
        B = torch.randn(K, N, device='cuda')
        bias = torch.randn(N, device='cuda')
        
        # PyTorch实现
        C_pytorch = torch.matmul(A, B) + bias
        
        # 自定义实现
        C_custom = custom_ops.gemm_bias(A, B, bias)
        
        # 计算误差
        max_diff = torch.max(torch.abs(C_custom - C_pytorch)).item()
        mean_diff = torch.mean(torch.abs(C_custom - C_pytorch)).item()
        relative_error = mean_diff / (torch.mean(torch.abs(C_pytorch)).item() + 1e-10)
        
        passed = relative_error < 1e-5
        status = "✓ 通过" if passed else "✗ 失败"
        
        print(f"  [{M}x{K}] @ [{K}x{N}] + bias: {status}")
        print(f"    最大误差: {max_diff:.2e}, 平均误差: {mean_diff:.2e}, 相对误差: {relative_error:.2e}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_gemm_bias_gelu():
    """测试GEMM+Bias+GELU正确性"""
    print("\n" + "=" * 60)
    print("测试GEMM+Bias+GELU正确性")
    print("=" * 60)
    
    test_cases = [
        (10, 20, 30),
        (128, 3072, 768),
        (512, 3072, 768),
    ]
    
    all_passed = True
    
    for M, N, K in test_cases:
        A = torch.randn(M, K, device='cuda')
        B = torch.randn(K, N, device='cuda')
        bias = torch.randn(N, device='cuda')
        
        # PyTorch实现
        C_pytorch = torch.nn.functional.gelu(torch.matmul(A, B) + bias)
        
        # 自定义实现
        C_custom = custom_ops.gemm_bias_gelu(A, B, bias)
        
        # 计算误差
        max_diff = torch.max(torch.abs(C_custom - C_pytorch)).item()
        mean_diff = torch.mean(torch.abs(C_custom - C_pytorch)).item()
        relative_error = mean_diff / (torch.mean(torch.abs(C_pytorch)).item() + 1e-10)
        
        # GELU是非线性激活，误差稍大一些也可以接受
        passed = relative_error < 1e-4
        status = "✓ 通过" if passed else "✗ 失败"
        
        print(f"  [{M}x{K}] @ [{K}x{N}] + bias + GELU: {status}")
        print(f"    最大误差: {max_diff:.2e}, 平均误差: {mean_diff:.2e}, 相对误差: {relative_error:.2e}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_layernorm():
    """测试LayerNorm正确性"""
    print("\n" + "=" * 60)
    print("测试LayerNorm正确性")
    print("=" * 60)
    
    test_cases = [
        (10, 768),
        (128, 768),
        (512, 768),
    ]
    
    all_passed = True
    
    for M, N in test_cases:
        x = torch.randn(M, N, device='cuda')
        gamma = torch.randn(N, device='cuda')
        beta = torch.randn(N, device='cuda')
        eps = 1e-12
        
        # PyTorch实现
        y_pytorch = torch.nn.functional.layer_norm(x, (N,), gamma, beta, eps)
        
        # 自定义实现
        y_custom = custom_ops.layernorm(x, gamma, beta, eps)
        
        # 计算误差
        max_diff = torch.max(torch.abs(y_custom - y_pytorch)).item()
        mean_diff = torch.mean(torch.abs(y_custom - y_pytorch)).item()
        relative_error = mean_diff / (torch.mean(torch.abs(y_pytorch)).item() + 1e-10)
        
        passed = relative_error < 1e-5
        status = "✓ 通过" if passed else "✗ 失败"
        
        print(f"  [{M}x{N}]: {status}")
        print(f"    最大误差: {max_diff:.2e}, 平均误差: {mean_diff:.2e}, 相对误差: {relative_error:.2e}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def main():
    print("=" * 60)
    print("BERT推理加速 - 正确性验证")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    
    # 运行所有测试
    results = []
    
    results.append(("GEMM", test_gemm()))
    results.append(("GEMM+Bias", test_gemm_bias()))
    results.append(("GEMM+Bias+GELU", test_gemm_bias_gelu()))
    results.append(("LayerNorm", test_layernorm()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败，请检查实现")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

