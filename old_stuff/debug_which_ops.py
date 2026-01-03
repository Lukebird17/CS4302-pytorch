"""
调试脚本：检查优化模型是否真的在使用自定义算子
"""

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from bert_optimized import create_optimized_bert

print("="*80)
print("调试：检查哪些自定义算子被调用")
print("="*80)

# 创建优化模型
model = create_optimized_bert("bert-base-uncased").cuda().eval()

# 准备输入
input_ids = torch.randint(0, 30522, (4, 128)).cuda()
attention_mask = torch.ones((4, 128)).cuda()

print("\n开始推理...")
with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask)

print("✓ 推理完成")

# 检查模型结构
print("\n" + "="*80)
print("模型结构分析")
print("="*80)

# 检查Linear层是否被替换
from bert_optimized import CustomLinear
import torch.nn as nn

linear_count = 0
custom_linear_count = 0

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        linear_count += 1
    elif isinstance(module, CustomLinear):
        custom_linear_count += 1

print(f"\n普通Linear层: {linear_count} 个")
print(f"CustomLinear层: {custom_linear_count} 个")

if custom_linear_count == 0:
    print("\n❌ 问题：没有任何CustomLinear层！")
    print("   原因：替换逻辑可能有问题")
else:
    print(f"\n✓ 成功替换了 {custom_linear_count} 个Linear层")

# 检查融合算子
from bert_optimized import FusedBertSelfOutput, FusedBertOutput
fused_count = 0

for name, module in model.named_modules():
    if isinstance(module, (FusedBertSelfOutput, FusedBertOutput)):
        fused_count += 1

print(f"融合LayerNorm模块: {fused_count} 个")

# 测试单独的GEMM性能
print("\n" + "="*80)
print("单独测试GEMM性能")
print("="*80)

A = torch.randn(4096, 768).cuda()
B = torch.randn(768, 768).cuda()

# 预热
for _ in range(10):
    _ = torch.matmul(A, B)
torch.cuda.synchronize()

# 测试PyTorch
import time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    C_pytorch = torch.matmul(A, B)
end.record()
torch.cuda.synchronize()
time_pytorch = start.elapsed_time(end) / 100

print(f"PyTorch GEMM (4096x768 @ 768x768): {time_pytorch:.3f} ms")

# 测试自定义GEMM
try:
    import bert_custom_gemm
    
    start.record()
    for _ in range(100):
        C_custom = bert_custom_gemm.custom_gemm(A, B)
    end.record()
    torch.cuda.synchronize()
    time_custom = start.elapsed_time(end) / 100
    
    print(f"Custom GEMM (4096x768 @ 768x768): {time_custom:.3f} ms")
    print(f"加速比: {time_pytorch / time_custom:.3f}x")
    
    # 验证正确性
    diff = torch.abs(C_pytorch - C_custom).max().item()
    print(f"最大误差: {diff:.6f}")
    
    if time_custom > time_pytorch:
        print("\n❌ 问题：自定义GEMM比PyTorch还慢！")
        print(f"   慢了 {time_custom / time_pytorch:.2f}x")
        print("   这就是为什么整体没有加速")
except ImportError:
    print("⚠ 自定义GEMM未编译")

print("\n" + "="*80)

