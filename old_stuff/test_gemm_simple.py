"""
简单测试：验证自定义GEMM的性能
对比cuBLAS vs 自定义GEMM
"""

import torch
import time
import bert_custom_gemm

device = torch.device('cuda')

# BERT典型的矩阵尺寸
# batch=4, seq=128, hidden=768 → M=512, K=768, N=768
M, K, N = 512, 768, 768

print("="*80)
print("GEMM性能对比测试")
print("="*80)
print(f"矩阵尺寸: M={M}, K={K}, N={N}")
print()

# 创建测试数据
A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)

# 预热
for _ in range(10):
    _ = torch.matmul(A, B)
    _ = bert_custom_gemm.custom_gemm(A, B)
torch.cuda.synchronize()

print("【测试1】cuBLAS (torch.matmul)")
times = []
for i in range(50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    C1 = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    
    times.append(start.elapsed_time(end))

import numpy as np
mean_cublas = np.mean(times)
std_cublas = np.std(times)
print(f"  平均耗时: {mean_cublas:.4f} ms (± {std_cublas:.4f} ms)")

print("\n【测试2】自定义GEMM")
times = []
for i in range(50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    C2 = bert_custom_gemm.custom_gemm(A, B)
    end.record()
    torch.cuda.synchronize()
    
    times.append(start.elapsed_time(end))

mean_custom = np.mean(times)
std_custom = np.std(times)
print(f"  平均耗时: {mean_custom:.4f} ms (± {std_custom:.4f} ms)")

# 验证正确性
print("\n【验证正确性】")
C1 = torch.matmul(A, B)
C2 = bert_custom_gemm.custom_gemm(A, B)
max_diff = torch.max(torch.abs(C1 - C2)).item()
rel_error = max_diff / torch.max(torch.abs(C1)).item()
print(f"  最大绝对误差: {max_diff:.6f}")
print(f"  相对误差: {rel_error:.6f}")

if rel_error < 1e-3:
    print("  ✓ 正确性验证通过")
else:
    print("  ✗ 正确性验证失败")

# 性能对比
print("\n" + "="*80)
speedup = mean_cublas / mean_custom
if speedup > 1:
    print(f"✓ 自定义GEMM加速 {speedup:.3f}x ({(speedup-1)*100:.1f}%)")
else:
    print(f"✗ 自定义GEMM变慢 {speedup:.3f}x ({(1-speedup)*100:.1f}%)")
print("="*80)

# 分析
print("\n提示:")
print("  - cuBLAS极度优化，达到50-80%性能已经很好")
print("  - 如果慢于cuBLAS 2倍以内是正常的")
print("  - 如果慢4倍以上，说明有严重问题")

