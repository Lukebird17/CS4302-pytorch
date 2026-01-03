"""
简单示例：展示如何使用自定义算子
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 尝试导入自定义算子
try:
    import custom_ops
    print("✓ 成功加载自定义算子")
    USE_CUSTOM = True
except ImportError:
    print("✗ 未能加载自定义算子，请先编译")
    USE_CUSTOM = False
    sys.exit(1)


def example_gemm():
    """GEMM示例"""
    print("\n" + "=" * 50)
    print("示例1: 基础GEMM操作")
    print("=" * 50)
    
    # 创建随机矩阵
    M, K, N = 100, 200, 150
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    
    # 使用自定义GEMM
    C = custom_ops.gemm(A, B, 1.0, 0.0)
    
    # 验证
    C_ref = torch.matmul(A, B)
    error = torch.max(torch.abs(C - C_ref)).item()
    
    print(f"矩阵大小: A[{M}x{K}] @ B[{K}x{N}] = C[{M}x{N}]")
    print(f"最大误差: {error:.2e}")
    print(f"结果: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def example_gemm_bias():
    """GEMM+Bias融合示例"""
    print("\n" + "=" * 50)
    print("示例2: GEMM+Bias融合")
    print("=" * 50)
    
    M, K, N = 100, 200, 150
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    bias = torch.randn(N, device='cuda')
    
    # 使用融合算子
    C = custom_ops.gemm_bias(A, B, bias)
    
    # 验证
    C_ref = torch.matmul(A, B) + bias
    error = torch.max(torch.abs(C - C_ref)).item()
    
    print(f"矩阵大小: A[{M}x{K}] @ B[{K}x{N}] + bias[{N}] = C[{M}x{N}]")
    print(f"最大误差: {error:.2e}")
    print(f"结果: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")
    
    print("\n优势: 将GEMM和Bias加法融合在一个kernel中")
    print("      - 减少内存读写")
    print("      - 降低kernel launch开销")


def example_gemm_bias_gelu():
    """GEMM+Bias+GELU融合示例"""
    print("\n" + "=" * 50)
    print("示例3: GEMM+Bias+GELU融合")
    print("=" * 50)
    
    M, K, N = 100, 200, 150
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    bias = torch.randn(N, device='cuda')
    
    # 使用融合算子
    C = custom_ops.gemm_bias_gelu(A, B, bias)
    
    # 验证
    C_ref = torch.nn.functional.gelu(torch.matmul(A, B) + bias)
    error = torch.max(torch.abs(C - C_ref)).item()
    
    print(f"矩阵大小: GELU(A[{M}x{K}] @ B[{K}x{N}] + bias[{N}]) = C[{M}x{N}]")
    print(f"最大误差: {error:.2e}")
    print(f"结果: {'✓ 通过' if error < 1e-3 else '✗ 失败'}")
    
    print("\n优势: 将GEMM、Bias和GELU激活融合在一个kernel中")
    print("      - 最大程度减少内存读写")
    print("      - 在BERT的FFN层中特别有效")


def example_layernorm():
    """LayerNorm示例"""
    print("\n" + "=" * 50)
    print("示例4: LayerNorm")
    print("=" * 50)
    
    M, N = 100, 768
    x = torch.randn(M, N, device='cuda')
    gamma = torch.ones(N, device='cuda')
    beta = torch.zeros(N, device='cuda')
    eps = 1e-12
    
    # 使用自定义LayerNorm
    y = custom_ops.layernorm(x, gamma, beta, eps)
    
    # 验证
    y_ref = torch.nn.functional.layer_norm(x, (N,), gamma, beta, eps)
    error = torch.max(torch.abs(y - y_ref)).item()
    
    print(f"输入大小: [{M}x{N}]")
    print(f"最大误差: {error:.2e}")
    print(f"结果: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")
    
    print("\n优势: 优化的LayerNorm实现")
    print("      - 使用warp-level reduce")
    print("      - 减少全局内存访问")


def example_bert_layer():
    """模拟BERT层的计算"""
    print("\n" + "=" * 50)
    print("示例5: 模拟BERT FFN层计算")
    print("=" * 50)
    
    # BERT base参数
    batch_size, seq_len = 32, 512
    hidden_size = 768
    intermediate_size = 3072
    
    # 模拟输入
    x = torch.randn(batch_size * seq_len, hidden_size, device='cuda')
    
    # FFN第一层的权重和偏置
    W1 = torch.randn(intermediate_size, hidden_size, device='cuda')
    b1 = torch.randn(intermediate_size, device='cuda')
    
    # FFN第二层的权重和偏置
    W2 = torch.randn(hidden_size, intermediate_size, device='cuda')
    b2 = torch.randn(hidden_size, device='cuda')
    
    print(f"输入: [{batch_size}x{seq_len}, {hidden_size}]")
    print(f"FFN第一层: [{hidden_size}] -> [{intermediate_size}] + GELU")
    print(f"FFN第二层: [{intermediate_size}] -> [{hidden_size}]")
    
    # 使用融合算子
    print("\n使用融合算子计算...")
    h1 = custom_ops.gemm_bias_gelu(x, W1.t(), b1)  # 第一层 + GELU
    h2 = custom_ops.gemm_bias(h1, W2.t(), b2)      # 第二层
    
    print(f"中间层形状: {h1.shape}")
    print(f"输出形状: {h2.shape}")
    print("✓ 完成BERT FFN层计算")
    
    print("\n性能优势:")
    print("  - 第一层: 3个操作融合为1个kernel (GEMM+Bias+GELU)")
    print("  - 第二层: 2个操作融合为1个kernel (GEMM+Bias)")
    print("  - 总共节省: 5个kernel -> 2个kernel")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持")
        sys.exit(1)
    
    print("=" * 50)
    print("自定义CUDA算子使用示例")
    print("=" * 50)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 运行所有示例
    example_gemm()
    example_gemm_bias()
    example_gemm_bias_gelu()
    example_layernorm()
    example_bert_layer()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)

