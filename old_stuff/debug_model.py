"""
调试：检查模型中实际使用的算子
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from transformers import BertModel
from bert_optimized import create_optimized_bert
import bert_custom_gemm

print("="*80)
print("模型调试")
print("="*80)

# 创建优化模型
model = create_optimized_bert("bert-base-uncased").cuda().eval()

# 创建输入
batch, seq = 4, 128
input_ids = torch.randint(0, 30522, (batch, seq)).cuda()
attention_mask = torch.ones((batch, seq)).cuda()

# 检查一下CustomLinear的权重形状
print("\n检查CustomLinear权重形状:")
for name, module in model.named_modules():
    if hasattr(module, 'weight_t'):
        print(f"  {name}: weight_t.shape = {module.weight_t.shape}")
        break

# 测试前向传播
print("\n执行前向传播...")
import time
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask)

torch.cuda.synchronize()
elapsed = (time.time() - start) * 1000
print(f"✓ 前向传播完成，耗时: {elapsed:.2f} ms")

# 对比baseline
print("\n对比Baseline模型...")
model_baseline = BertModel.from_pretrained("bert-base-uncased").cuda().eval()

torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    output_baseline = model_baseline(input_ids=input_ids, attention_mask=attention_mask)

torch.cuda.synchronize()
elapsed_baseline = (time.time() - start) * 1000
print(f"✓ Baseline完成，耗时: {elapsed_baseline:.2f} ms")

print("\n" + "="*80)
print(f"优化模型: {elapsed:.2f} ms")
print(f"Baseline: {elapsed_baseline:.2f} ms")
print(f"加速比: {elapsed_baseline/elapsed:.3f}x")
print("="*80)

