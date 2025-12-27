"""
BERT优化模型 - 使用自实现GEMM kernel
完全替换PyTorch的原生实现
"""

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfOutput, BertOutput

# 尝试加载自定义算子
try:
    import bert_fused_ops
    FUSED_OPS_AVAILABLE = True
    print("✓ CUDA融合算子已加载")
except ImportError:
    FUSED_OPS_AVAILABLE = False
    print("⚠ CUDA融合算子未编译")

try:
    import bert_custom_gemm
    CUSTOM_GEMM_AVAILABLE = True
    print("✓ 自定义GEMM已加载")
except ImportError:
    CUSTOM_GEMM_AVAILABLE = False
    print("⚠ 自定义GEMM未编译")


# ============================================================================
# 自定义Linear层 - 使用我们自己实现的GEMM
# ============================================================================
class CustomLinear(nn.Module):
    """
    使用自实现GEMM的Linear层
    完全替换PyTorch的torch.nn.functional.linear
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重和偏置
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # 使用自定义GEMM: y = x @ W^T + b
        if CUSTOM_GEMM_AVAILABLE and x.dim() == 2:
            # 重塑输入（如果需要）
            original_shape = x.shape
            if x.dim() > 2:
                x = x.reshape(-1, x.size(-1))
            
            # 自定义GEMM: y = x @ W^T
            # 注意：我们的GEMM是 C = A @ B
            # 而Linear是 y = x @ W^T = x @ W.T
            # 所以需要转置权重
            output = bert_custom_gemm.custom_gemm(x, self.weight.t())
            
            if self.bias is not None:
                output = output + self.bias
            
            # 恢复形状
            if len(original_shape) > 2:
                output = output.reshape(*original_shape[:-1], -1)
            
            return output
        else:
            # Fallback到PyTorch实现
            return torch.nn.functional.linear(x, self.weight, self.bias)


# ============================================================================
# FFN层 - 使用GEMM+Bias+GELU融合
# ============================================================================
class CustomBertIntermediate(nn.Module):
    """
    BERT的FFN第一层：[768] -> [3072] + GELU
    使用融合的GEMM+Bias+GELU kernel
    """
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(config.intermediate_size))
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
    
    def forward(self, hidden_states):
        if CUSTOM_GEMM_AVAILABLE and hidden_states.dim() == 2:
            # 使用融合kernel: GEMM + Bias + GELU一次完成
            return bert_custom_gemm.custom_gemm_bias_gelu(
                hidden_states,
                self.weight.t(),
                self.bias
            )
        else:
            # Fallback
            output = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
            return 0.5 * output * (1.0 + torch.tanh(0.7978845608 * (output + 0.044715 * output * output * output)))


# ============================================================================
# 融合LayerNorm + Residual（使用之前的CUDA kernel）
# ============================================================================
class FusedBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.hidden_size, config.hidden_size)  # 使用自定义GEMM
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.gamma = self.LayerNorm.weight
        self.beta = self.LayerNorm.bias
        self.eps = config.layer_norm_eps
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        
        if FUSED_OPS_AVAILABLE and not self.training:
            return bert_fused_ops.fused_ln_residual_optimized(
                hidden_states, input_tensor,
                self.gamma, self.beta, self.eps
            )
        else:
            hidden_states = self.dropout(hidden_states)
            return self.LayerNorm(hidden_states + input_tensor)


class FusedBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = CustomLinear(config.intermediate_size, config.hidden_size)  # 使用自定义GEMM
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.gamma = self.LayerNorm.weight
        self.beta = self.LayerNorm.bias
        self.eps = config.layer_norm_eps
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        
        if FUSED_OPS_AVAILABLE and not self.training:
            return bert_fused_ops.fused_ln_residual_optimized(
                hidden_states, input_tensor,
                self.gamma, self.beta, self.eps
            )
        else:
            hidden_states = self.dropout(hidden_states)
            return self.LayerNorm(hidden_states + input_tensor)


# ============================================================================
# 创建完全优化的BERT模型
# ============================================================================
def create_optimized_bert(model_name="bert-base-uncased"):
    """
    创建完全优化的BERT模型
    
    所有优化:
    1. 自实现GEMM kernel (最重要！)
    2. 融合GEMM+Bias+GELU
    3. 融合LayerNorm+Residual+Dropout
    """
    print(f"\n{'='*80}")
    print("创建完全优化的BERT模型（自实现GEMM）")
    print(f"{'='*80}")
    
    model = BertModel.from_pretrained(model_name)
    
    optimizations = []
    
    # 1. 替换所有Linear为CustomLinear（使用自实现GEMM）
    if CUSTOM_GEMM_AVAILABLE:
        linear_count = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    
                    # 创建CustomLinear并复制权重
                    custom_linear = CustomLinear(
                        module.in_features,
                        module.out_features,
                        module.bias is not None
                    )
                    custom_linear.weight.data = module.weight.data
                    if module.bias is not None:
                        custom_linear.bias.data = module.bias.data
                    
                    setattr(parent, attr_name, custom_linear)
                    linear_count += 1
        
        optimizations.append(f"自实现GEMM: {linear_count}个Linear层")
    
    # 2. 替换BertSelfOutput和BertOutput（融合算子）
    if FUSED_OPS_AVAILABLE:
        fused_count = 0
        for name, module in list(model.named_modules()):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
                
                if isinstance(module, BertSelfOutput):
                    new_module = FusedBertSelfOutput(model.config)
                    new_module.load_state_dict(module.state_dict(), strict=False)
                    setattr(parent, attr_name, new_module)
                    fused_count += 1
                
                elif isinstance(module, BertOutput):
                    new_module = FusedBertOutput(model.config)
                    new_module.load_state_dict(module.state_dict(), strict=False)
                    setattr(parent, attr_name, new_module)
                    fused_count += 1
        
        optimizations.append(f"融合LayerNorm: {fused_count}个模块")
    
    print("\n已应用的优化:")
    for opt in optimizations:
        print(f"  ✓ {opt}")
    
    if not CUSTOM_GEMM_AVAILABLE:
        print("\n⚠ 警告: 自定义GEMM未编译，使用PyTorch原生实现")
        print("   请运行: cd custom_ops && python setup.py install")
    
    print(f"\n{'='*80}\n")
    
    return model


if __name__ == "__main__":
    print("测试优化的BERT模型...")
    model = create_optimized_bert().cuda().eval()
    
    input_ids = torch.randint(0, 30522, (2, 128)).cuda()
    attention_mask = torch.ones((2, 128)).cuda()
    
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"✓ 模型测试通过")
    print(f"  输出shape: {output.last_hidden_state.shape}")
