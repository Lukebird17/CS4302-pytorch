"""
测试自定义GEMM算子的正确性
"""
import torch
import bert_custom_gemm
import time

print("="*80)
print("测试自定义GEMM算子")
print("="*80)

# 测试1: 小规模矩阵
print("\n【测试1】小规模矩阵 (256x768) @ (768x768)")
M, K, N = 256, 768, 768
A = torch.randn(M, K).cuda()
B = torch.randn(K, N).cuda()

# PyTorch标准结果
C_torch = torch.matmul(A, B)

# 自定义GEMM结果
C_custom = bert_custom_gemm.custom_gemm(A, B)

# 比较结果
diff = torch.abs(C_torch - C_custom)
max_diff = torch.max(diff).item()
mean_diff = torch.mean(diff).item()
rel_error = torch.mean(diff / (torch.abs(C_torch) + 1e-8)).item()

print(f"  最大误差: {max_diff:.6f}")
print(f"  平均误差: {mean_diff:.6f}")
print(f"  相对误差: {rel_error:.6f}")

if max_diff < 1e-2:
    print("  ✓ 测试通过！")
else:
    print("  ❌ 测试失败，误差过大")
    exit(1)

# 测试2: BERT典型大小
print("\n【测试2】BERT典型大小 (512x768) @ (768x3072)")
M, K, N = 512, 768, 3072
A = torch.randn(M, K).cuda()
B = torch.randn(K, N).cuda()

C_torch = torch.matmul(A, B)
C_custom = bert_custom_gemm.custom_gemm(A, B)

diff = torch.abs(C_torch - C_custom)
max_diff = torch.max(diff).item()
rel_error = torch.mean(diff / (torch.abs(C_torch) + 1e-8)).item()

print(f"  最大误差: {max_diff:.6f}")
print(f"  相对误差: {rel_error:.6f}")

if max_diff < 1e-2:
    print("  ✓ 测试通过！")
else:
    print("  ❌ 测试失败")
    exit(1)

# 测试3: 融合GEMM+Bias+GELU
print("\n【测试3】GEMM+Bias+GELU融合")
M, K, N = 512, 768, 3072
A = torch.randn(M, K).cuda()
B = torch.randn(K, N).cuda()
bias = torch.randn(N).cuda()

# PyTorch参考实现
def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

C_torch = gelu(torch.matmul(A, B) + bias)
C_custom = bert_custom_gemm.custom_gemm_bias_gelu(A, B, bias)

diff = torch.abs(C_torch - C_custom)
max_diff = torch.max(diff).item()
rel_error = torch.mean(diff / (torch.abs(C_torch) + 1e-8)).item()

print(f"  最大误差: {max_diff:.6f}")
print(f"  相对误差: {rel_error:.6f}")

if max_diff < 1e-2:
    print("  ✓ 测试通过！")
else:
    print("  ❌ 测试失败")
    exit(1)

# 测试4: 性能benchmark
print("\n【测试4】性能对比")
M, K, N = 1024, 768, 768
A = torch.randn(M, K).cuda()
B = torch.randn(K, N).cuda()

# 预热
for _ in range(10):
    _ = torch.matmul(A, B)
    _ = bert_custom_gemm.custom_gemm(A, B)
torch.cuda.synchronize()

# PyTorch
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    _ = torch.matmul(A, B)
end.record()
torch.cuda.synchronize()
time_torch = start.elapsed_time(end) / 100

# Custom GEMM
start.record()
for _ in range(100):
    _ = bert_custom_gemm.custom_gemm(A, B)
end.record()
torch.cuda.synchronize()
time_custom = start.elapsed_time(end) / 100

print(f"  PyTorch matmul: {time_torch:.3f} ms")
print(f"  Custom GEMM:    {time_custom:.3f} ms")
print(f"  相对性能:       {time_torch/time_custom:.2f}x")

if time_custom < time_torch * 2:
    print("  ✓ 性能可接受（在cuBLAS 2倍范围内）")
else:
    print("  ⚠ 性能较慢，但功能正常")

print("\n" + "="*80)
print("✓ 所有测试通过！GEMM算子工作正常。")
print("="*80)

