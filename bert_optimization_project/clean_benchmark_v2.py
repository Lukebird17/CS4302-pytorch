"""
干净的性能对比测试 - 不使用hooks
严格控制测试条件，多轮测试取平均
"""

import torch
import numpy as np
from transformers import BertModel
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from bert_optimized import create_optimized_bert

print("="*80)
print("BERT 算子优化 - 性能对比测试")
print("="*80)

if not torch.cuda.is_available():
    print("需要CUDA支持")
    exit(1)

device = torch.device('cuda')

# ============================================================================
# 创建模型（无hooks）
# ============================================================================
print("\n【第1步】加载模型")
print("-"*80)

print("加载Baseline模型...")
model_baseline = BertModel.from_pretrained("bert-base-uncased").cuda()
model_baseline.eval()

print("加载优化模型...")
model_opt = create_optimized_bert("bert-base-uncased").cuda()
model_opt.eval()

# ============================================================================
# 测试配置
# ============================================================================
TEST_CONFIGS = [
    {'batch': 1, 'seq': 128},
    {'batch': 4, 'seq': 128},
    {'batch': 8, 'seq': 128},
    {'batch': 16, 'seq': 128},
    {'batch': 32, 'seq': 128},
]

NUM_ROUNDS = 5  # 测试轮数
ITERS_PER_ROUND = 100  # 每轮迭代次数

print(f"\n【第2步】测试配置")
print("-"*80)
print(f"测试轮数: {NUM_ROUNDS}")
print(f"每轮迭代: {ITERS_PER_ROUND}")
print(f"配置数量: {len(TEST_CONFIGS)}")

# ============================================================================
# 执行测试
# ============================================================================
print(f"\n【第3步】执行性能测试")
print("-"*80)

all_results = []

for config in TEST_CONFIGS:
    batch_size = config['batch']
    seq_len = config['seq']
    
    print(f"\n测试配置: batch={batch_size}, seq={seq_len}")
    print("  " + "-"*60)
    
    # 准备输入
    input_ids = torch.randint(0, 30522, (batch_size, seq_len)).cuda()
    attention_mask = torch.ones((batch_size, seq_len)).cuda()
    
    # 预热
    print("  预热中...", end='', flush=True)
    for _ in range(20):
        with torch.no_grad():
            _ = model_baseline(input_ids=input_ids, attention_mask=attention_mask)
            _ = model_opt(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    print(" 完成")
    
    # 多轮测试
    baseline_times = []
    optimized_times = []
    
    for round_idx in range(NUM_ROUNDS):
        # 测试 Baseline
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(ITERS_PER_ROUND):
                _ = model_baseline(input_ids=input_ids, attention_mask=attention_mask)
        end_event.record()
        torch.cuda.synchronize()
        
        time_baseline = start_event.elapsed_time(end_event) / ITERS_PER_ROUND
        baseline_times.append(time_baseline)
        
        # 测试 Optimized
        start_event.record()
        with torch.no_grad():
            for _ in range(ITERS_PER_ROUND):
                _ = model_opt(input_ids=input_ids, attention_mask=attention_mask)
        end_event.record()
        torch.cuda.synchronize()
        
        time_opt = start_event.elapsed_time(end_event) / ITERS_PER_ROUND
        optimized_times.append(time_opt)
        
        speedup = time_baseline / time_opt
        improvement = (1 - time_opt / time_baseline) * 100
        
        print(f"  轮次{round_idx+1}: Baseline={time_baseline:.2f}ms, "
              f"Optimized={time_opt:.2f}ms, "
              f"加速比={speedup:.3f}x ({improvement:+.1f}%)")
    
    # 统计分析
    baseline_times = np.array(baseline_times)
    optimized_times = np.array(optimized_times)
    
    mean_baseline = np.mean(baseline_times)
    mean_optimized = np.mean(optimized_times)
    std_baseline = np.std(baseline_times)
    std_optimized = np.std(optimized_times)
    
    mean_speedup = mean_baseline / mean_optimized
    mean_improvement = (1 - mean_optimized / mean_baseline) * 100
    
    cv_baseline = std_baseline / mean_baseline * 100
    cv_optimized = std_optimized / mean_optimized * 100
    
    print(f"\n  统计结果:")
    print(f"    Baseline:  {mean_baseline:.2f} ± {std_baseline:.2f} ms (CV={cv_baseline:.1f}%)")
    print(f"    Optimized: {mean_optimized:.2f} ± {std_optimized:.2f} ms (CV={cv_optimized:.1f}%)")
    print(f"    平均加速比: {mean_speedup:.3f}x")
    print(f"    平均性能变化: {mean_improvement:+.1f}%")
    
    all_results.append({
        'batch_size': batch_size,
        'seq_len': seq_len,
        'baseline_mean': mean_baseline,
        'baseline_std': std_baseline,
        'optimized_mean': mean_optimized,
        'optimized_std': std_optimized,
        'speedup': mean_speedup,
        'improvement': mean_improvement,
        'cv_baseline': cv_baseline,
        'cv_optimized': cv_optimized
    })

# ============================================================================
# 汇总结果
# ============================================================================
print("\n" + "="*80)
print("【第4步】测试结果汇总")
print("="*80)

print(f"\n{'Batch':<8} {'Seq':<6} {'Baseline(ms)':<14} {'Optimized(ms)':<14} {'加速比':<10} {'性能变化':<10}")
print("-"*80)

for result in all_results:
    print(f"{result['batch_size']:<8} "
          f"{result['seq_len']:<6} "
          f"{result['baseline_mean']:>6.2f}±{result['baseline_std']:<5.2f} "
          f"{result['optimized_mean']:>6.2f}±{result['optimized_std']:<5.2f} "
          f"{result['speedup']:>8.3f}x "
          f"{result['improvement']:>+8.1f}%")

# 计算总体改进
overall_improvement = np.mean([r['improvement'] for r in all_results])
print("\n" + "="*80)
print(f"总体平均性能变化: {overall_improvement:+.1f}%")
print("="*80)

# 稳定性检查
unstable_tests = [r for r in all_results if r['cv_baseline'] > 5 or r['cv_optimized'] > 5]
if unstable_tests:
    print(f"\n⚠ 警告: {len(unstable_tests)} 个测试的变异系数 > 5%，结果可能不稳定")
else:
    print(f"\n✓ 所有测试的变异系数 < 5%，结果稳定可靠")

if overall_improvement > 3:
    print(f"\n✓ V2优化版本实现了明显的性能提升！")
elif overall_improvement > 0:
    print(f"\n⚠ V2优化版本有小幅性能提升")
else:
    print(f"\n❌ V2优化版本性能未提升，需要进一步优化")
    print("   可能原因：")
    print("   - PyTorch原生实现使用了cuDNN等高度优化的库")
    print("   - kernel实现仍有优化空间")
    print("   - 小batch size下kernel启动开销占比大")

print("\n" + "="*80)

