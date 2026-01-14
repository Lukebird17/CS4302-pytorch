import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import sys
import os

# 导入自定义算子
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../custom_ops'))
try:
    import custom_ops
    USE_CUSTOM_OPS = True
    print("成功加载自定义CUDA算子")
except ImportError:
    USE_CUSTOM_OPS = False
    print("警告: 无法加载自定义算子，将使用PyTorch默认实现")


class OptimizedLinear(nn.Module):
    """使用自定义GEMM的优化线性层"""
    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.activation = activation
        
        # 初始化
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # x: [batch_size, seq_len, in_features] 或 [batch_size*seq_len, in_features]
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, self.in_features)
        
        if USE_CUSTOM_OPS:
            # 使用自定义融合算子
            if self.activation == 'gelu' and self.bias is not None:
                output = custom_ops.gemm_bias_gelu(x, self.weight.t(), self.bias)
            elif self.bias is not None:
                output = custom_ops.gemm_bias(x, self.weight.t(), self.bias)
            else:
                output = custom_ops.gemm(x, self.weight.t(), 1.0, 0.0)
                if self.activation == 'gelu':
                    output = torch.nn.functional.gelu(output)
        else:
            # 回退到PyTorch实现
            output = torch.nn.functional.linear(x, self.weight, self.bias)
            if self.activation == 'gelu':
                output = torch.nn.functional.gelu(output)
        
        if len(original_shape) == 3:
            output = output.reshape(original_shape[0], original_shape[1], -1)
        
        return output


class OptimizedLayerNorm(nn.Module):
    """使用自定义算子的优化LayerNorm"""
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, hidden_size = x.shape
            x = x.reshape(-1, hidden_size)
        
        if USE_CUSTOM_OPS:
            output = custom_ops.layernorm(x, self.weight, self.bias, self.eps)
        else:
            output = torch.nn.functional.layer_norm(
                x, (x.shape[-1],), self.weight, self.bias, self.eps
            )
        
        if len(original_shape) == 3:
            output = output.reshape(original_shape)
        
        return output


class OptimizedBertSelfAttention(nn.Module):
    """优化的BERT自注意力层"""
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = OptimizedLinear(config.hidden_size, self.all_head_size)
        self.key = OptimizedLinear(config.hidden_size, self.all_head_size)
        self.value = OptimizedLinear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer


class OptimizedBertSelfOutput(nn.Module):
    """优化的BERT自注意力输出层"""
    def __init__(self, config):
        super().__init__()
        self.dense = OptimizedLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = OptimizedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class OptimizedBertAttention(nn.Module):
    """优化的BERT注意力模块"""
    def __init__(self, config):
        super().__init__()
        self.self_attention = OptimizedBertSelfAttention(config)
        self.output = OptimizedBertSelfOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self_attention(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class OptimizedBertIntermediate(nn.Module):
    """优化的BERT中间层（使用融合算子）"""
    def __init__(self, config):
        super().__init__()
        self.dense = OptimizedLinear(
            config.hidden_size, 
            config.intermediate_size,
            activation='gelu'
        )
    
    def forward(self, hidden_states):
        return self.dense(hidden_states)


class OptimizedBertOutput(nn.Module):
    """优化的BERT输出层"""
    def __init__(self, config):
        super().__init__()
        self.dense = OptimizedLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = OptimizedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class OptimizedBertLayer(nn.Module):
    """优化的BERT层"""
    def __init__(self, config):
        super().__init__()
        self.attention = OptimizedBertAttention(config)
        self.intermediate = OptimizedBertIntermediate(config)
        self.output = OptimizedBertOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class OptimizedBertEncoder(nn.Module):
    """优化的BERT编码器"""
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([OptimizedBertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class OptimizedBertModel(nn.Module):
    """优化的BERT模型，用于分类任务"""
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super().__init__()
        
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        # 创建优化的编码器
        self.optimized_encoder = OptimizedBertEncoder(self.config)
        
        # 分类头
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = OptimizedLinear(self.config.hidden_size, num_labels)
        
        # 复制权重
        self._copy_weights()
    
    def _copy_weights(self):
        """将预训练权重复制到优化的模块"""
        with torch.no_grad():
            for i, layer in enumerate(self.optimized_encoder.layer):
                orig_layer = self.bert.encoder.layer[i]
                
                # 复制attention权重
                layer.attention.self_attention.query.weight.copy_(
                    orig_layer.attention.self.query.weight
                )
                layer.attention.self_attention.query.bias.copy_(
                    orig_layer.attention.self.query.bias
                )
                layer.attention.self_attention.key.weight.copy_(
                    orig_layer.attention.self.key.weight
                )
                layer.attention.self_attention.key.bias.copy_(
                    orig_layer.attention.self.key.bias
                )
                layer.attention.self_attention.value.weight.copy_(
                    orig_layer.attention.self.value.weight
                )
                layer.attention.self_attention.value.bias.copy_(
                    orig_layer.attention.self.value.bias
                )
                
                # 复制attention输出权重
                layer.attention.output.dense.weight.copy_(
                    orig_layer.attention.output.dense.weight
                )
                layer.attention.output.dense.bias.copy_(
                    orig_layer.attention.output.dense.bias
                )
                layer.attention.output.LayerNorm.weight.copy_(
                    orig_layer.attention.output.LayerNorm.weight
                )
                layer.attention.output.LayerNorm.bias.copy_(
                    orig_layer.attention.output.LayerNorm.bias
                )
                
                # 复制中间层权重
                layer.intermediate.dense.weight.copy_(
                    orig_layer.intermediate.dense.weight
                )
                layer.intermediate.dense.bias.copy_(
                    orig_layer.intermediate.dense.bias
                )
                
                # 复制输出层权重
                layer.output.dense.weight.copy_(
                    orig_layer.output.dense.weight
                )
                layer.output.dense.bias.copy_(
                    orig_layer.output.dense.bias
                )
                layer.output.LayerNorm.weight.copy_(
                    orig_layer.output.LayerNorm.weight
                )
                layer.output.LayerNorm.bias.copy_(
                    orig_layer.output.LayerNorm.bias
                )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 使用原始BERT的embedding层
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        # 准备attention mask
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # 使用优化的编码器
        encoder_output = self.optimized_encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        
        # 池化和分类
        pooled_output = encoder_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


def create_optimized_bert_model(model_name='bert-base-uncased', num_labels=2):
    """创建优化的BERT模型"""
    model = OptimizedBertModel(model_name, num_labels)
    return model




