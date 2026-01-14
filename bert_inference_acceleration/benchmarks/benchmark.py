"""
性能测试和对比脚本
对比自定义GEMM算子 vs PyTorch原生实现
"""
import torch
import time
import sys
import os
import argparse
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import custom_ops
    USE_CUSTOM_OPS = True
except ImportError:
    USE_CUSTOM_OPS = False
    print("警告: 自定义算子未编译")


def benchmark_gemm(M, N, K, num_iters=100, warmup=10, device='cuda'):
    """测试GEMM性能"""
    print(f"\n测试GEMM: M={M}, N={N}, K={K}")
    
    # 生成随机矩阵
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    
    results = {}
    
    # 1. PyTorch原生实现
    print("  测试PyTorch原生matmul...")
    for _ in range(warmup):
        _ = torch.matmul(A, B)
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    results['pytorch'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000
    }
    
    # 2. 自定义GEMM
    if USE_CUSTOM_OPS:
        print("  测试自定义GEMM...")
        for _ in range(warmup):
            _ = custom_ops.gemm(A, B, 1.0, 0.0)
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            C_custom = custom_ops.gemm(A, B, 1.0, 0.0)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        results['custom'] = {
            'mean': np.mean(times) * 1000,
            'std': np.std(times) * 1000,
            'min': np.min(times) * 1000,
            'max': np.max(times) * 1000
        }
        
        # 验证正确性
        C_pytorch = torch.matmul(A, B)
        max_diff = torch.max(torch.abs(C_custom - C_pytorch)).item()
        mean_diff = torch.mean(torch.abs(C_custom - C_pytorch)).item()
        results['correctness'] = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'relative_error': mean_diff / torch.mean(torch.abs(C_pytorch)).item()
        }
    
    return results


def benchmark_gemm_bias(M, N, K, num_iters=100, warmup=10, device='cuda'):
    """测试GEMM+Bias性能"""
    print(f"\n测试GEMM+Bias: M={M}, N={N}, K={K}")
    
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    bias = torch.randn(N, device=device)
    
    results = {}
    
    # PyTorch原生实现
    print("  测试PyTorch原生实现...")
    for _ in range(warmup):
        _ = torch.matmul(A, B) + bias
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        C = torch.matmul(A, B) + bias
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    results['pytorch'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000
    }
    
    # 自定义融合算子
    if USE_CUSTOM_OPS:
        print("  测试自定义融合算子...")
        for _ in range(warmup):
            _ = custom_ops.gemm_bias(A, B, bias)
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            C_custom = custom_ops.gemm_bias(A, B, bias)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        results['custom_fused'] = {
            'mean': np.mean(times) * 1000,
            'std': np.std(times) * 1000
        }
        
        # 验证正确性
        C_pytorch = torch.matmul(A, B) + bias
        max_diff = torch.max(torch.abs(C_custom - C_pytorch)).item()
        results['max_diff'] = max_diff
    
    return results


def benchmark_gemm_bias_gelu(M, N, K, num_iters=100, warmup=10, device='cuda'):
    """测试GEMM+Bias+GELU性能"""
    print(f"\n测试GEMM+Bias+GELU: M={M}, N={N}, K={K}")
    
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    bias = torch.randn(N, device=device)
    
    results = {}
    
    # PyTorch原生实现
    print("  测试PyTorch原生实现...")
    for _ in range(warmup):
        _ = torch.nn.functional.gelu(torch.matmul(A, B) + bias)
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        C = torch.nn.functional.gelu(torch.matmul(A, B) + bias)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    results['pytorch'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000
    }
    
    # 自定义融合算子
    if USE_CUSTOM_OPS:
        print("  测试自定义融合算子...")
        for _ in range(warmup):
            _ = custom_ops.gemm_bias_gelu(A, B, bias)
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            C_custom = custom_ops.gemm_bias_gelu(A, B, bias)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        results['custom_fused'] = {
            'mean': np.mean(times) * 1000,
            'std': np.std(times) * 1000
        }
        
        # 验证正确性
        C_pytorch = torch.nn.functional.gelu(torch.matmul(A, B) + bias)
        max_diff = torch.max(torch.abs(C_custom - C_pytorch)).item()
        results['max_diff'] = max_diff
    
    return results


def main():
    parser = argparse.ArgumentParser(description='性能测试和对比')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--num_iters', type=int, default=100, help='测试迭代次数')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    print("=" * 80)
    print("BERT推理加速 - 性能测试")
    print("=" * 80)
    print(f"设备: {args.device}")
    print(f"测试迭代次数: {args.num_iters}")
    print(f"预热次数: {args.warmup}")
    
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("=" * 80)
    
    # BERT相关的典型矩阵大小
    # batch_size=32, seq_len=512, hidden_size=768, intermediate_size=3072
    test_cases = [
        # (M, N, K) - 对应不同的BERT操作
        (32, 768, 768),      # Self-attention Q/K/V投影 (batch*seq_len较小时)
        (128, 768, 768),     # Self-attention Q/K/V投影 (batch*seq_len中等时)
        (512, 768, 768),     # Self-attention Q/K/V投影 (batch*seq_len较大时)
        (32, 3072, 768),     # FFN第一层
        (128, 3072, 768),    # FFN第一层
        (512, 3072, 768),    # FFN第一层
        (32, 768, 3072),     # FFN第二层
        (128, 768, 3072),    # FFN第二层
        (512, 768, 3072),    # FFN第二层
    ]
    
    all_results = []
    
    print("\n" + "=" * 80)
    print("1. GEMM性能测试")
    print("=" * 80)
    
    for M, N, K in test_cases[:3]:  # 只测试几个典型大小
        results = benchmark_gemm(M, N, K, args.num_iters, args.warmup, args.device)
        
        row = [f"{M}x{K}x{N}"]
        if 'pytorch' in results:
            row.append(f"{results['pytorch']['mean']:.3f}±{results['pytorch']['std']:.3f}")
        else:
            row.append("N/A")
        
        if 'custom' in results:
            row.append(f"{results['custom']['mean']:.3f}±{results['custom']['std']:.3f}")
            speedup = results['pytorch']['mean'] / results['custom']['mean']
            row.append(f"{speedup:.2f}x")
            if 'correctness' in results:
                row.append(f"{results['correctness']['relative_error']:.2e}")
        else:
            row.extend(["N/A", "N/A", "N/A"])
        
        all_results.append(row)
    
    print("\n" + tabulate(
        all_results,
        headers=["矩阵大小", "PyTorch (ms)", "自定义GEMM (ms)", "加速比", "相对误差"],
        tablefmt="grid"
    ))
    
    print("\n" + "=" * 80)
    print("2. GEMM+Bias融合测试")
    print("=" * 80)
    
    fusion_results = []
    for M, N, K in [(128, 768, 768), (128, 3072, 768)]:
        results = benchmark_gemm_bias(M, N, K, args.num_iters, args.warmup, args.device)
        
        row = [f"{M}x{K}x{N}"]
        row.append(f"{results['pytorch']['mean']:.3f}±{results['pytorch']['std']:.3f}")
        
        if 'custom_fused' in results:
            row.append(f"{results['custom_fused']['mean']:.3f}±{results['custom_fused']['std']:.3f}")
            speedup = results['pytorch']['mean'] / results['custom_fused']['mean']
            row.append(f"{speedup:.2f}x")
            row.append(f"{results['max_diff']:.2e}")
        else:
            row.extend(["N/A", "N/A", "N/A"])
        
        fusion_results.append(row)
    
    print("\n" + tabulate(
        fusion_results,
        headers=["矩阵大小", "PyTorch (ms)", "融合算子 (ms)", "加速比", "最大误差"],
        tablefmt="grid"
    ))
    
    print("\n" + "=" * 80)
    print("3. GEMM+Bias+GELU融合测试")
    print("=" * 80)
    
    gelu_results = []
    for M, N, K in [(128, 3072, 768)]:
        results = benchmark_gemm_bias_gelu(M, N, K, args.num_iters, args.warmup, args.device)
        
        row = [f"{M}x{K}x{N}"]
        row.append(f"{results['pytorch']['mean']:.3f}±{results['pytorch']['std']:.3f}")
        
        if 'custom_fused' in results:
            row.append(f"{results['custom_fused']['mean']:.3f}±{results['custom_fused']['std']:.3f}")
            speedup = results['pytorch']['mean'] / results['custom_fused']['mean']
            row.append(f"{speedup:.2f}x")
            row.append(f"{results['max_diff']:.2e}")
        else:
            row.extend(["N/A", "N/A", "N/A"])
        
        gelu_results.append(row)
    
    print("\n" + tabulate(
        gelu_results,
        headers=["矩阵大小", "PyTorch (ms)", "融合算子 (ms)", "加速比", "最大误差"],
        tablefmt="grid"
    ))
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()




