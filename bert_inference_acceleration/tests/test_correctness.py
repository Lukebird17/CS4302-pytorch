"""
BERT 推理加速 - 正确性验证 (完全对齐性能脚本布局)
修复说明：
1. 权重定义为 (N, K) 以模拟 nn.Linear.weight
2. 调用时使用 weight.t().contiguous() 以传入 (K, N) 布局
"""
import torch
import sys
import os
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import custom_ops
    print("✓ 成功加载自定义算子")
except ImportError:
    print("❌ 加载失败，请检查编译")
    sys.exit(1)

def test_gemm():
    print("\n" + "=" * 60)
    print("测试 GEMM 正确性 (模拟 Linear 布局)")
    print("=" * 60)
    
    # M: batch*seq, N: out_features, K: in_features
    test_cases = [
        (128, 768, 768),    # BERT Attention
        (512, 3072, 768),   # BERT FFN Up-projection
        (512, 768, 3072),   # BERT FFN Down-projection
    ]
    
    all_passed = True
    for M, N, K in test_cases:
        input_tensor = torch.randn(M, K, device='cuda')
        # 严格对齐 PyTorch Linear: weight 形状是 (out, in)
        weight = torch.randn(N, K, device='cuda')
        
        # PyTorch 参考: x * weight^T
        C_pytorch = F.linear(input_tensor, weight)
        
        # 自定义算子: 传入转置后的 (K, N) 连续内存
        C_custom = custom_ops.gemm(input_tensor, weight.t().contiguous(), 1.0, 0.0)
        
        # 使用L2范数（Frobenius范数）计算误差
        diff_l2 = torch.norm(C_custom - C_pytorch, p='fro').item()
        ref_l2 = torch.norm(C_pytorch, p='fro').item()
        rel_error = diff_l2 / (ref_l2 + 1e-8)  # 避免除零
        
        passed = rel_error < 1e-4
        print(f"    差异L2范数: {diff_l2:.2e}, 参考L2范数: {ref_l2:.2e}, 相对误差: {rel_error:.2e}")
        print(f"  [{M}x{K}] @ [{K}x{N}]: {'✓ 通过' if passed else '✗ 失败'}")
        if not passed:
            print(f"    差异L2范数: {diff_l2:.2e}, 参考L2范数: {ref_l2:.2e}, 相对误差: {rel_error:.2e}")
            all_passed = False
    return all_passed

def test_gemm_bias_gelu():
    print("\n" + "=" * 60)
    print("测试 GEMM+Bias+GELU 融合算子")
    print("=" * 60)
    
    M, N, K = 512, 3072, 768
    input_tensor = torch.randn(M, K, device='cuda')
    weight = torch.randn(N, K, device='cuda')
    bias = torch.randn(N, device='cuda')
    
    # PyTorch 参考
    C_pytorch = F.gelu(F.linear(input_tensor, weight, bias), approximate='tanh')
    
    # 自定义融合算子
    C_custom = custom_ops.gemm_bias_gelu(input_tensor, weight.t().contiguous(), bias)
    
    # 使用L2范数（Frobenius范数）计算误差
    diff_l2 = torch.norm(C_custom - C_pytorch, p='fro').item()
    ref_l2 = torch.norm(C_pytorch, p='fro').item()
    rel_error = diff_l2 / (ref_l2 + 1e-8)  # 避免除零
    
    passed = rel_error < 1e-4
    print(f"    差异L2范数: {diff_l2:.2e}, 参考L2范数: {ref_l2:.2e}, 相对误差: {rel_error:.2e}")
    print(f"  [{M}x{K}] + Bias + GELU: {'✓ 通过' if passed else '✗ 失败'}")
    if not passed:
        print(f"    差异L2范数: {diff_l2:.2e}, 参考L2范数: {ref_l2:.2e}, 相对误差: {rel_error:.2e}")
    return passed

def test_layernorm():
    print("\n" + "=" * 60)
    print("测试 LayerNorm 正确性")
    print("=" * 60)
    
    M, N = 512, 768
    x = torch.randn(M, N, device='cuda')
    gamma = torch.ones(N, device='cuda')
    beta = torch.zeros(N, device='cuda')
    
    y_pytorch = F.layer_norm(x, (N,), gamma, beta, 1e-12)
    y_custom = custom_ops.layernorm(x, gamma, beta, 1e-12)
    
    # 使用L2范数（Frobenius范数）计算误差
    diff_l2 = torch.norm(y_custom - y_pytorch, p='fro').item()
    ref_l2 = torch.norm(y_pytorch, p='fro').item()
    rel_error = diff_l2 / (ref_l2 + 1e-8)  # 避免除零
    
    passed = rel_error < 1e-5
    print(f"    差异L2范数: {diff_l2:.2e}, 参考L2范数: {ref_l2:.2e}, 相对误差: {rel_error:.2e}")
    print(f"  [{M}x{N}]: {'✓ 通过' if passed else '✗ 失败'}")
    if not passed:
        print(f"    差异L2范数: {diff_l2:.2e}, 参考L2范数: {ref_l2:.2e}, 相对误差: {rel_error:.2e}")
    return passed

def main():
    if not torch.cuda.is_available(): return
    
    results = [
        ("GEMM", test_gemm()),
        ("GEMM_BIAS_GELU", test_gemm_bias_gelu()),
        ("LayerNorm", test_layernorm())
    ]
    
    print("\n" + "=" * 60)
    all_ok = all(r[1] for r in results)
    if all_ok:
        print("✅ 所有针对 BERT 场景的算子验证通过！")
    else:
        print("❌ 部分测试失败，请检查内存对齐或索引逻辑")
    print("=" * 60)

if __name__ == '__main__':
    main()