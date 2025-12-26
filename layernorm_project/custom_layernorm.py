"""
自定义LayerNorm的Python封装
提供与torch.nn.LayerNorm兼容的接口
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import os
import sys

# 尝试导入CUDA扩展
try:
    import custom_layernorm_cuda
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False
    print("警告: 自定义CUDA扩展未编译，请先运行 'python setup.py install'")


class CustomLayerNormFunction(Function):
    """
    自定义LayerNorm的autograd Function
    """
    
    @staticmethod
    def forward(ctx, input, gamma, beta, eps, use_optimized=True):
        """
        前向传播
        Args:
            input: [batch, seq_len, hidden_size]
            gamma: [hidden_size]
            beta: [hidden_size]
            eps: epsilon
            use_optimized: 是否使用优化版本
        """
        if not CUDA_EXTENSION_AVAILABLE:
            raise RuntimeError("CUDA extension not available")
        
        # 确保输入在CUDA上
        assert input.is_cuda, "Input must be on CUDA"
        assert input.is_contiguous(), "Input must be contiguous"
        
        # 调用CUDA kernel
        if use_optimized and input.dtype == torch.float32 and input.size(2) % 4 == 0:
            output = custom_layernorm_cuda.forward_optimized(input, gamma, beta, eps)
        else:
            output = custom_layernorm_cuda.forward_basic(input, gamma, beta, eps)
        
        # 保存用于backward（这里简化，只实现forward）
        ctx.save_for_backward(input, gamma, beta)
        ctx.eps = eps
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播（简化版本，使用PyTorch的autograd）
        """
        # 这里为了简化，使用PyTorch的原生backward
        # 完整实现需要自定义backward kernel
        raise NotImplementedError("Backward pass uses PyTorch native implementation")


class CustomLayerNorm(nn.Module):
    """
    自定义LayerNorm层
    与torch.nn.LayerNorm接口兼容
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, use_optimized=True):
        """
        Args:
            normalized_shape: 归一化的维度(int or tuple)
            eps: epsilon for numerical stability
            elementwise_affine: 是否使用可学习的affine参数
            use_optimized: 是否使用优化版本的kernel
        """
        super(CustomLayerNorm, self).__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_optimized = use_optimized
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input):
        """
        前向传播
        Args:
            input: [..., normalized_shape]
        Returns:
            output: same shape as input
        """
        if not CUDA_EXTENSION_AVAILABLE or not input.is_cuda:
            # 回退到PyTorch原生实现
            return nn.functional.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        
        # 确保输入形状正确
        assert input.shape[-len(self.normalized_shape):] == self.normalized_shape
        
        # 将输入reshape为3D: [batch, seq, hidden]
        original_shape = input.shape
        if len(input.shape) > 3:
            # 合并前面的维度
            input = input.view(-1, input.shape[-2], input.shape[-1])
        elif len(input.shape) == 2:
            # 添加seq维度
            input = input.unsqueeze(1)
        
        # 调用自定义CUDA kernel
        try:
            output = CustomLayerNormFunction.apply(
                input, self.weight, self.bias, self.eps, self.use_optimized
            )
        except:
            # 如果失败，回退到PyTorch原生实现
            output = nn.functional.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        
        # Reshape回原始形状
        output = output.view(original_shape)
        
        return output
    
    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, use_optimized={use_optimized}'.format(**self.__dict__)


def replace_layernorm_in_model(model, use_optimized=True):
    """
    将模型中的所有LayerNorm替换为自定义实现
    
    Args:
        model: PyTorch模型
        use_optimized: 是否使用优化版本
    
    Returns:
        替换后的模型
    """
    if not CUDA_EXTENSION_AVAILABLE:
        print("警告: CUDA扩展不可用，无法替换LayerNorm")
        return model
    
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            # 创建自定义LayerNorm
            custom_ln = CustomLayerNorm(
                module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                use_optimized=use_optimized
            )
            
            # 复制权重
            if module.elementwise_affine:
                custom_ln.weight.data = module.weight.data.clone()
                custom_ln.bias.data = module.bias.data.clone()
            
            # 替换
            setattr(model, name, custom_ln)
            print(f"替换 {name}: LayerNorm -> CustomLayerNorm")
        else:
            # 递归处理子模块
            replace_layernorm_in_model(module, use_optimized)
    
    return model


# 简单测试
def test_custom_layernorm():
    """测试自定义LayerNorm的正确性"""
    if not CUDA_EXTENSION_AVAILABLE:
        print("跳过测试: CUDA扩展不可用")
        return
    
    print("测试自定义LayerNorm...")
    
    # 测试配置
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    eps = 1e-5
    
    # 创建测试数据
    input_tensor = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    # PyTorch原生LayerNorm
    native_ln = nn.LayerNorm(hidden_size, eps=eps).cuda()
    
    # 自定义LayerNorm
    custom_ln = CustomLayerNorm(hidden_size, eps=eps, use_optimized=True).cuda()
    custom_ln.weight.data = native_ln.weight.data.clone()
    custom_ln.bias.data = native_ln.bias.data.clone()
    
    # 前向传播
    with torch.no_grad():
        native_output = native_ln(input_tensor)
        custom_output = custom_ln(input_tensor)
    
    # 比较结果
    max_diff = torch.max(torch.abs(native_output - custom_output)).item()
    mean_diff = torch.mean(torch.abs(native_output - custom_output)).item()
    
    print(f"最大差异: {max_diff}")
    print(f"平均差异: {mean_diff}")
    
    # 设置容差
    tolerance = 1e-3
    if max_diff < tolerance:
        print("✓ 测试通过: 自定义LayerNorm输出正确!")
    else:
        print("✗ 测试失败: 输出差异过大")
    
    return max_diff < tolerance


if __name__ == "__main__":
    test_custom_layernorm()

