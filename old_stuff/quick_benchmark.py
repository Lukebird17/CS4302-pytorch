"""
快速性能测试：验证新GEMM优化
"""
import torch
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

print("="*80)
print("BERT GEMM性能测试 - 验证优化效果")
print("="*80)

# 导入自定义GEMM
try:
    import bert_custom_gemm
    print("✓ 自定义GEMM已加载")
except ImportError as e:
    print(f"❌ 无法加载自定义GEMM: {e}")
    exit(1)

device = torch.device('cuda')

# ============================================================================
# 测试1: 单独GEMM性能对比
# ============================================================================
print("\n" + "="*80)
print("【测试1】单独GEMM性能 - 典型BERT矩阵尺寸")
print("="*80)

test_configs = [
    # (M, K, N, description)
    (512, 768, 768, "QKV投影: [512, 768] @ [768, 768]"),
    (4096, 768, 768, "大batch QKV: [4096, 768] @ [768, 768]"),
    (512, 768, 3072, "FFN第1层: [512, 768] @ [768, 3072]"),
    (4096, 768, 3072, "大batch FFN: [4096, 768] @ [768, 3072]"),
    (512, 3072, 768, "FFN第2层: [512, 3072] @ [3072, 768]"),
]

for M, K, N, desc in test_configs:
    print(f"\n{desc}")
    print("-" * 60)
    
    # 准备数据
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # 预热
    for _ in range(10):
        _ = torch.matmul(A, B)
        _ = bert_custom_gemm.custom_gemm(A, B)
    torch.cuda.synchronize()
    
    # 测试PyTorch (cuBLAS)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        C_pytorch = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    time_pytorch = start.elapsed_time(end) / 100
    
    # 测试自定义GEMM
    start.record()
    for _ in range(100):
        C_custom = bert_custom_gemm.custom_gemm(A, B)
    end.record()
    torch.cuda.synchronize()
    time_custom = start.elapsed_time(end) / 100
    
    # 验证正确性
    diff = torch.abs(C_pytorch - C_custom).max().item()
    
    # 输出结果
    speedup = time_pytorch / time_custom
    improvement = (1 - time_custom / time_pytorch) * 100
    
    print(f"  cuBLAS时间:     {time_pytorch:.4f} ms")
    print(f"  自定义GEMM:     {time_custom:.4f} ms")
    print(f"  加速比:         {speedup:.3f}x ({improvement:+.1f}%)")
    print(f"  最大误差:       {diff:.2e}")
    
    if speedup > 0.9:
        print(f"  ✓ 性能优秀！达到cuBLAS的{speedup*100:.1f}%")
    elif speedup > 0.7:
        print(f"  ✓ 性能良好，达到cuBLAS的{speedup*100:.1f}%")
    elif speedup > 0.5:
        print(f"  ⚠ 性能一般，仅达到cuBLAS的{speedup*100:.1f}%")
    else:
        print(f"  ❌ 性能较差，仅达到cuBLAS的{speedup*100:.1f}%")

# ============================================================================
# 测试2: 完整BERT模型性能
# ============================================================================
print("\n" + "="*80)
print("【测试2】完整BERT模型性能对比")
print("="*80)

from bert_optimized import create_optimized_bert
from transformers import BertModel

# 加载模型
print("\n加载模型...")
model_baseline = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
model_optimized = create_optimized_bert("bert-base-uncased").cuda().eval()

# 测试配置
test_cases = [
    {'batch': 1, 'seq': 128, 'desc': '单样本推理'},
    {'batch': 8, 'seq': 128, 'desc': '小批量'},
    {'batch': 32, 'seq': 128, 'desc': '中批量'},
]

print(f"\n开始测试...")
for config in test_cases:
    batch = config['batch']
    seq = config['seq']
    desc = config['desc']
    
    print(f"\n{desc}: batch={batch}, seq={seq}")
    print("-" * 60)
    
    input_ids = torch.randint(0, 30522, (batch, seq)).cuda()
    attention_mask = torch.ones((batch, seq)).cuda()
    
    # 预热
    for _ in range(5):
        with torch.no_grad():
            _ = model_baseline(input_ids=input_ids, attention_mask=attention_mask)
            _ = model_optimized(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    
    # 测试Baseline
    start.record()
    with torch.no_grad():
        for _ in range(50):
            _ = model_baseline(input_ids=input_ids, attention_mask=attention_mask)
    end.record()
    torch.cuda.synchronize()
    time_baseline = start.elapsed_time(end) / 50
    
    # 测试Optimized
    start.record()
    with torch.no_grad():
        for _ in range(50):
            _ = model_optimized(input_ids=input_ids, attention_mask=attention_mask)
    end.record()
    torch.cuda.synchronize()
    time_opt = start.elapsed_time(end) / 50
    
    speedup = time_baseline / time_opt
    improvement = (1 - time_opt / time_baseline) * 100
    
    print(f"  Baseline:   {time_baseline:.3f} ms")
    print(f"  Optimized:  {time_opt:.3f} ms")
    print(f"  加速比:     {speedup:.3f}x ({improvement:+.1f}%)")
    
    if improvement > 15:
        print(f"  ✓✓ 优化效果显著！")
    elif improvement > 8:
        print(f"  ✓ 优化效果良好")
    elif improvement > 3:
        print(f"  ⚠ 优化效果一般")
    else:
        print(f"  ❌ 优化效果不明显")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("测试完成！")
print("="*80)
print("\n关键指标:")
print("  1. 如果单独GEMM达到cuBLAS 70-90%性能 → 实现成功")
print("  2. 如果整体模型加速10-20% → 优化显著")
print("  3. 融合算子(LayerNorm+GELU)贡献额外3-5%加速")
print("\n" + "="*80)

