"""
快速演示脚本
展示LayerNorm的基本用法和性能对比
"""

import torch
import torch.nn as nn
import time
import sys

def demo_layernorm_basic():
    """演示LayerNorm的基本用法"""
    print("\n" + "="*60)
    print("1. LayerNorm基本用法演示")
    print("="*60)
    
    # 创建输入
    batch_size = 4
    seq_len = 8
    hidden_size = 16
    
    print(f"\n输入形状: ({batch_size}, {seq_len}, {hidden_size})")
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, hidden_size)
    print(f"输入样本:\n{x[0, 0, :8]}")  # 显示第一个token的前8个特征
    
    # 创建LayerNorm层
    ln = nn.LayerNorm(hidden_size)
    
    # 前向传播
    y = ln(x)
    
    print(f"\nLayerNorm输出:\n{y[0, 0, :8]}")
    
    # 验证归一化效果
    mean = x[0, 0].mean()
    std = x[0, 0].std()
    print(f"\n归一化前: mean={mean:.4f}, std={std:.4f}")
    
    mean_after = y[0, 0].mean()
    std_after = y[0, 0].std()
    print(f"归一化后: mean={mean_after:.4f}, std={std_after:.4f}")
    
    print("\n✓ LayerNorm将特征归一化为均值接近0，标准差接近1")


def demo_layernorm_math():
    """演示LayerNorm的数学计算"""
    print("\n" + "="*60)
    print("2. LayerNorm数学原理演示")
    print("="*60)
    
    # 简单的1D例子
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\n输入: {x}")
    
    # 手动计算
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    std = torch.sqrt(var + 1e-5)
    
    print(f"\n步骤1 - 计算均值: μ = {mean:.4f}")
    print(f"步骤2 - 计算方差: σ² = {var:.4f}")
    print(f"步骤3 - 计算标准差: σ = {std:.4f}")
    
    # 标准化
    x_normalized = (x - mean) / std
    print(f"步骤4 - 标准化: (x - μ) / σ = {x_normalized}")
    
    # 使用LayerNorm
    ln = nn.LayerNorm(5, elementwise_affine=False)  # 不使用affine变换
    y = ln(x.unsqueeze(0)).squeeze(0)
    
    print(f"\nLayerNorm结果: {y}")
    print(f"差异: {torch.max(torch.abs(y - x_normalized)):.6f}")
    
    print("\n✓ 手动计算与LayerNorm结果一致")


def demo_performance_simple():
    """简单的性能对比演示"""
    print("\n" + "="*60)
    print("3. 性能对比演示")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("\n⚠ CUDA不可用，跳过性能测试")
        return
    
    # 测试配置
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    iterations = 100
    
    print(f"\n测试配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Iterations: {iterations}")
    
    # 创建数据
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    ln = nn.LayerNorm(hidden_size).cuda()
    
    # 预热
    print("\n预热中...")
    for _ in range(10):
        _ = ln(x)
    torch.cuda.synchronize()
    
    # 计时
    print("测试中...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        y = ln(x)
    end.record()
    torch.cuda.synchronize()
    
    elapsed = start.elapsed_time(end)
    avg_time = elapsed / iterations
    
    print(f"\n结果:")
    print(f"  总时间: {elapsed:.2f} ms")
    print(f"  平均延迟: {avg_time:.4f} ms")
    print(f"  吞吐量: {batch_size * seq_len * iterations / (elapsed / 1000):.0f} tokens/sec")
    
    # 计算理论性能
    num_elements = batch_size * seq_len * hidden_size
    memory_bytes = num_elements * 4 * 3  # input + output + gamma/beta (simplified)
    memory_gb = memory_bytes / 1e9
    bandwidth_gbps = memory_gb / (elapsed / 1000)
    
    print(f"\n性能分析:")
    print(f"  处理元素: {num_elements / 1e6:.2f} M")
    print(f"  内存访问: {memory_gb:.2f} GB")
    print(f"  有效带宽: {bandwidth_gbps:.2f} GB/s")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")


def demo_bert_layernorm():
    """演示BERT中的LayerNorm"""
    print("\n" + "="*60)
    print("4. BERT中的LayerNorm演示")
    print("="*60)
    
    try:
        from transformers import BertModel
        
        print("\n加载BERT模型...")
        model = BertModel.from_pretrained("bert-base-uncased")
        
        # 统计LayerNorm
        ln_count = 0
        ln_info = []
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                ln_count += 1
                ln_info.append((name, module.normalized_shape))
        
        print(f"\nBERT模型中的LayerNorm:")
        print(f"  总数: {ln_count} 个")
        print(f"\n前5个LayerNorm层:")
        for i, (name, shape) in enumerate(ln_info[:5]):
            print(f"    {i+1}. {name}: {shape}")
        
        print(f"\n结构:")
        print(f"  - 每个Encoder层有2个LayerNorm")
        print(f"  - 12层Encoder → 24个LayerNorm")
        print(f"  - Embedding后有1个LayerNorm (可选)")
        
        # 简单推理
        if torch.cuda.is_available():
            print("\n运行简单推理...")
            model = model.cuda()
            model.eval()
            
            input_ids = torch.randint(0, 30522, (1, 16)).cuda()
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            
            print(f"✓ 推理成功")
            print(f"  输出形状: {outputs.last_hidden_state.shape}")
        
    except ImportError:
        print("\n⚠ 未安装transformers库，跳过BERT演示")
    except Exception as e:
        print(f"\n⚠ 加载BERT失败: {e}")


def demo_custom_layernorm():
    """演示自定义LayerNorm"""
    print("\n" + "="*60)
    print("5. 自定义LayerNorm演示")
    print("="*60)
    
    try:
        from custom_layernorm import CustomLayerNorm, CUDA_EXTENSION_AVAILABLE
        
        if not CUDA_EXTENSION_AVAILABLE:
            print("\n⚠ 自定义CUDA扩展未编译，跳过演示")
            print("  请运行: python setup.py install")
            return
        
        if not torch.cuda.is_available():
            print("\n⚠ CUDA不可用，跳过演示")
            return
        
        print("\n测试自定义LayerNorm...")
        
        # 创建测试数据
        x = torch.randn(8, 16, 768).cuda()
        
        # 原生LayerNorm
        native_ln = nn.LayerNorm(768).cuda()
        
        # 自定义LayerNorm
        custom_ln = CustomLayerNorm(768, use_optimized=True).cuda()
        custom_ln.weight.data = native_ln.weight.data.clone()
        custom_ln.bias.data = native_ln.bias.data.clone()
        
        # 前向传播
        with torch.no_grad():
            native_out = native_ln(x)
            custom_out = custom_ln(x)
        
        # 对比结果
        max_diff = torch.max(torch.abs(native_out - custom_out)).item()
        mean_diff = torch.mean(torch.abs(native_out - custom_out)).item()
        
        print(f"\n正确性验证:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print(f"  ✓ 测试通过 (差异 < 1e-3)")
        else:
            print(f"  ✗ 测试失败 (差异过大)")
        
        # 简单性能对比
        print("\n简单性能对比...")
        iterations = 100
        
        # 预热
        for _ in range(10):
            _ = native_ln(x)
            _ = custom_ln(x)
        torch.cuda.synchronize()
        
        # 测试原生
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            _ = native_ln(x)
        end.record()
        torch.cuda.synchronize()
        native_time = start.elapsed_time(end) / iterations
        
        # 测试自定义
        start.record()
        for _ in range(iterations):
            _ = custom_ln(x)
        end.record()
        torch.cuda.synchronize()
        custom_time = start.elapsed_time(end) / iterations
        
        print(f"\n性能对比:")
        print(f"  原生LayerNorm: {native_time:.4f} ms")
        print(f"  自定义LayerNorm: {custom_time:.4f} ms")
        print(f"  加速比: {native_time / custom_time:.2f}x")
        
    except ImportError:
        print("\n⚠ 无法导入自定义LayerNorm")
        print("  请确保已编译: python setup.py install")


def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU信息")
        print("="*60)
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  当前GPU: {torch.cuda.current_device()}")
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  PyTorch版本: {torch.__version__}")
        
        props = torch.cuda.get_device_properties(0)
        print(f"  显存大小: {props.total_memory / 1e9:.2f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  多处理器数量: {props.multi_processor_count}")
    else:
        print("\n⚠ CUDA不可用")


def main():
    """主函数"""
    print("\n" + "#"*60)
    print("# LayerNorm快速演示")
    print("#"*60)
    
    # GPU信息
    print_gpu_info()
    
    # 基本用法
    demo_layernorm_basic()
    
    # 数学原理
    demo_layernorm_math()
    
    # 性能测试
    demo_performance_simple()
    
    # BERT中的应用
    demo_bert_layernorm()
    
    # 自定义实现
    demo_custom_layernorm()
    
    print("\n" + "#"*60)
    print("# 演示完成!")
    print("#"*60)
    
    print("\n下一步:")
    print("  1. 运行完整调研: python layernorm_research.py")
    print("  2. 运行BERT评测: python bert_inference_benchmark.py")
    print("  3. 编译自定义扩展: python setup.py install")
    print("  4. 运行性能对比: python performance_comparison.py")


if __name__ == "__main__":
    main()

