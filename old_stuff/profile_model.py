"""
性能分析：找出到底哪里慢了
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from transformers import BertModel
from bert_optimized import create_optimized_bert
import bert_custom_gemm

# 创建模型
print("加载优化模型...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_opt = create_optimized_bert("bert-base-uncased").cuda().eval()

# 创建输入
batch, seq = 4, 128
input_ids = torch.randint(0, 30522, (batch, seq)).cuda()
attention_mask = torch.ones((batch, seq)).cuda()

print("\n使用PyTorch Profiler分析性能...")
print("="*80)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=False,
) as prof:
    with torch.no_grad():
        for _ in range(10):
            output = model_opt(input_ids=input_ids, attention_mask=attention_mask)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 查找自定义GEMM调用
print("\n" + "="*80)
print("自定义GEMM调用统计:")
print("="*80)
gemm_events = [evt for evt in prof.key_averages() if 'gemm' in evt.key.lower() or 'matmul' in evt.key.lower()]
for evt in gemm_events[:10]:
    print(f"{evt.key[:60]:60s} {evt.cuda_time_total/1000:>10.2f} ms  {evt.count:>6d} calls")

