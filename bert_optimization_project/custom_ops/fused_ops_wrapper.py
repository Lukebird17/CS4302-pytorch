"""
优化算子的Python Wrapper - 最终版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入CUDA扩展
FUSED_OPS_AVAILABLE = False
try:
    import bert_fused_ops
    FUSED_OPS_AVAILABLE = True
    print("✓ 优化融合算子已加载")
except ImportError:
    print("⚠ 优化融合算子不可用，将使用原生PyTorch实现")


class OptimizedFusedLayerNormResidual(nn.Module):
    """
    优化的融合LayerNorm+Residual
    
    优化点:
    - Welford一遍扫描算法
    - 针对hidden_size=768特化
    - eval模式专用（无dropout）
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, input, residual):
        if not FUSED_OPS_AVAILABLE or not input.is_cuda:
            x = input + residual
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        return bert_fused_ops.fused_ln_residual_optimized(
            input.contiguous(),
            residual.contiguous(),
            self.weight,
            self.bias,
            self.eps
        )


class FastGELU(nn.Module):
    """
    快速GELU（tanh近似 + 向量化）
    
    GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        if not FUSED_OPS_AVAILABLE or not input.is_cuda:
            return F.gelu(input, approximate='tanh')
        
        return bert_fused_ops.fast_gelu(input.contiguous())


class OptimizedSoftmax(nn.Module):
    """
    优化的Softmax（Online算法 + seq_len=128特化）
    """
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, input):
        if not FUSED_OPS_AVAILABLE or not input.is_cuda or self.dim != -1:
            return F.softmax(input, dim=self.dim)
        
        # 假设输入是 [batch*heads, seq, seq]
        return bert_fused_ops.optimized_softmax(input.contiguous())


def bias_gelu_fusion_inplace(input, bias):
    """
    原地执行 Bias + GELU 融合
    """
    if FUSED_OPS_AVAILABLE and input.is_cuda:
        bert_fused_ops.bias_gelu_fusion(input, bias)
    else:
        input.add_(bias.unsqueeze(0))
        input.copy_(F.gelu(input, approximate='tanh'))
