"""
融合算子使用示例
演示如何在实际模型中使用自定义融合算子
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 配置库路径
os.environ['LD_LIBRARY_PATH'] = os.path.join(
    os.path.dirname(torch.__file__), 'lib'
) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    from custom_ops_cuda import (
        gemm_bias_add_layernorm,
        gemm_bias_gelu_add_layernorm
    )
    print("✅ 成功导入自定义算子")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先运行: bash install.sh")
    sys.exit(1)


class OptimizedBertAttentionOutput(nn.Module):
    """
    使用融合算子优化的 BERT Attention 输出层
    替代原生的 Linear + Add + LayerNorm
    """
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.use_fused = True  # 是否使用融合算子
        
    def forward(self, hidden_states, input_tensor):
        """
        Args:
            hidden_states: [batch_size * seq_len, hidden_size]
            input_tensor: [batch_size * seq_len, hidden_size] (残差)
        Returns:
            [batch_size * seq_len, hidden_size]
        """
        if self.use_fused and hidden_states.is_cuda:
            # 融合算子路径 (5 操作 → 1 操作)
            hidden_states = gemm_bias_add_layernorm(
                hidden_states,                           # 输入
                self.dense.weight.t().contiguous(),     # 权重（需要转置）
                self.dense.bias,                         # Bias
                input_tensor,                            # 残差
                self.LayerNorm.weight,                   # LayerNorm gamma
                self.LayerNorm.bias,                     # LayerNorm beta
                self.LayerNorm.eps                       # epsilon
            )
        else:
            # PyTorch 原生路径
            hidden_states = self.dense(hidden_states)
            hidden_states = hidden_states + input_tensor
            hidden_states = self.LayerNorm(hidden_states)
        
        return self.dropout(hidden_states)


class OptimizedBertFFNOutput(nn.Module):
    """
    使用融合算子优化的 BERT FFN 输出层
    替代原生的 Linear + GELU + Add + LayerNorm
    """
    def __init__(self, intermediate_size, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.use_fused = True
        
    def forward(self, hidden_states, input_tensor):
        """
        Args:
            hidden_states: [batch_size * seq_len, intermediate_size]
            input_tensor: [batch_size * seq_len, hidden_size] (残差)
        Returns:
            [batch_size * seq_len, hidden_size]
        """
        if self.use_fused and hidden_states.is_cuda:
            # 融合算子路径 (6 操作 → 1 操作)
            # 注意：这里假设 GELU 已经在前一层应用
            # 如果需要在这一层应用 GELU，使用 gemm_bias_gelu_add_layernorm
            hidden_states = gemm_bias_add_layernorm(
                hidden_states,
                self.dense.weight.t().contiguous(),
                self.dense.bias,
                input_tensor,
                self.LayerNorm.weight,
                self.LayerNorm.bias,
                self.LayerNorm.eps
            )
        else:
            # PyTorch 原生路径
            hidden_states = self.dense(hidden_states)
            hidden_states = hidden_states + input_tensor
            hidden_states = self.LayerNorm(hidden_states)
        
        return self.dropout(hidden_states)


def demo_attention_output():
    """演示 Attention 输出层的使用"""
    print("\n" + "="*70)
    print("示例 1: Attention 输出层")
    print("="*70)
    
    batch_size = 16
    seq_len = 128
    hidden_size = 768
    
    # 创建模型
    model = OptimizedBertAttentionOutput(hidden_size).cuda()
    model.eval()
    
    # 准备输入
    hidden_states = torch.randn(batch_size * seq_len, hidden_size).cuda()
    input_tensor = torch.randn(batch_size * seq_len, hidden_size).cuda()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(hidden_states, input_tensor)
    torch.cuda.synchronize()
    
    # 测试融合算子
    model.use_fused = True
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        output_fused = model(hidden_states, input_tensor)
    end.record()
    torch.cuda.synchronize()
    time_fused = start.elapsed_time(end)
    
    # 测试原生实现
    model.use_fused = False
    torch.cuda.synchronize()
    
    start.record()
    with torch.no_grad():
        output_native = model(hidden_states, input_tensor)
    end.record()
    torch.cuda.synchronize()
    time_native = start.elapsed_time(end)
    
    # 验证正确性
    max_diff = torch.max(torch.abs(output_fused - output_native)).item()
    mean_diff = torch.mean(torch.abs(output_fused - output_native)).item()
    
    print(f"输入形状: [{batch_size}, {seq_len}, {hidden_size}]")
    print(f"\n性能对比:")
    print(f"  PyTorch 原生: {time_native:.4f} ms")
    print(f"  融合算子:     {time_fused:.4f} ms")
    print(f"  加速比:       {time_native/time_fused:.2f}x")
    print(f"\n正确性验证:")
    print(f"  最大误差: {max_diff:.2e}")
    print(f"  平均误差: {mean_diff:.2e}")
    print(f"  状态: {'✅ 通过' if max_diff < 1e-4 else '❌ 失败'}")


def demo_ffn_with_gelu():
    """演示 FFN 层（包含 GELU）的使用"""
    print("\n" + "="*70)
    print("示例 2: FFN 层 (带 GELU 融合)")
    print("="*70)
    
    batch_size = 16
    seq_len = 128
    hidden_size = 768
    intermediate_size = 3072
    
    # 准备输入
    hidden_states = torch.randn(batch_size * seq_len, intermediate_size).cuda()
    input_tensor = torch.randn(batch_size * seq_len, hidden_size).cuda()
    
    # 创建权重和参数
    weight = torch.randn(hidden_size, intermediate_size).cuda()
    bias = torch.randn(hidden_size).cuda()
    gamma = torch.ones(hidden_size).cuda()
    beta = torch.zeros(hidden_size).cuda()
    
    # 预热
    for _ in range(10):
        _ = torch.nn.functional.linear(hidden_states, weight, bias)
    torch.cuda.synchronize()
    
    # 测试融合算子（包含 GELU）
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        output_fused = gemm_bias_gelu_add_layernorm(
            hidden_states,
            weight.t().contiguous(),
            bias,
            input_tensor,
            gamma,
            beta,
            1e-12
        )
    end.record()
    torch.cuda.synchronize()
    time_fused = start.elapsed_time(end)
    
    # 测试原生实现
    start.record()
    with torch.no_grad():
        x = torch.nn.functional.linear(hidden_states, weight, bias)
        x = torch.nn.functional.gelu(x)
        x = x + input_tensor
        output_native = torch.nn.functional.layer_norm(x, (hidden_size,), gamma, beta, 1e-12)
    end.record()
    torch.cuda.synchronize()
    time_native = start.elapsed_time(end)
    
    # 验证正确性
    max_diff = torch.max(torch.abs(output_fused - output_native)).item()
    mean_diff = torch.mean(torch.abs(output_fused - output_native)).item()
    
    print(f"输入形状: [{batch_size}, {seq_len}, {intermediate_size}] → [{batch_size}, {seq_len}, {hidden_size}]")
    print(f"\n性能对比:")
    print(f"  PyTorch 原生: {time_native:.4f} ms")
    print(f"  融合算子:     {time_fused:.4f} ms")
    print(f"  加速比:       {time_native/time_fused:.2f}x")
    print(f"\n正确性验证:")
    print(f"  最大误差: {max_diff:.2e}")
    print(f"  平均误差: {mean_diff:.2e}")
    print(f"  状态: {'✅ 通过' if max_diff < 1e-4 else '❌ 失败'}")


def demo_basic_usage():
    """演示基础的算子调用"""
    print("\n" + "="*70)
    print("示例 3: 基础算子调用")
    print("="*70)
    
    M, N, K = 128, 768, 768
    
    # 准备输入
    A = torch.randn(M, K).cuda()
    B = torch.randn(N, K).cuda()  # 注意：这里是 (N, K)
    bias = torch.randn(N).cuda()
    residual = torch.randn(M, N).cuda()
    gamma = torch.ones(N).cuda()
    beta = torch.zeros(N).cuda()
    
    print(f"矩阵尺寸: A={A.shape}, B={B.shape}")
    print(f"计算: C = Linear(A, B^T) + bias + residual")
    print(f"      C = LayerNorm(C)")
    
    # 调用融合算子
    with torch.no_grad():
        output = gemm_bias_add_layernorm(
            A,                      # [M, K]
            B.t().contiguous(),    # [K, N] - 传入时需要转置
            bias,                   # [N]
            residual,              # [M, N]
            gamma,                 # [N]
            beta,                  # [N]
            1e-12                  # eps
        )
    
    print(f"输出形状: {output.shape}")
    print(f"✅ 算子调用成功")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ 需要 CUDA 支持")
        sys.exit(1)
    
    print("="*70)
    print("BERT 融合算子使用示例")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    try:
        # 运行示例
        demo_basic_usage()
        demo_attention_output()
        demo_ffn_with_gelu()
        
        print("\n" + "="*70)
        print("✅ 所有示例运行成功！")
        print("="*70)
        print("\n提示:")
        print("  - 融合算子适用于 batch_size >= 8 的场景")
        print("  - 在完整 BERT 模型中效果更明显")
        print("  - 确保输入数据在 GPU 上")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
