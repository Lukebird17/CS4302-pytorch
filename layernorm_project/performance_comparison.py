"""
性能对比测试脚本
对比PyTorch原生LayerNorm和自定义优化LayerNorm的性能
"""

import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

# 尝试导入自定义LayerNorm
try:
    from custom_layernorm import CustomLayerNorm, CUDA_EXTENSION_AVAILABLE
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False
    print("警告: 无法导入自定义LayerNorm")


class LayerNormPerformanceComparison:
    """LayerNorm性能对比工具"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
        
        self.results = {}
    
    def warmup(self, model, input_tensor, iterations=10):
        """GPU预热"""
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    def benchmark_single_config(self, batch_size, seq_len, hidden_size, iterations=1000):
        """
        单个配置的性能测试
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            hidden_size: 隐藏层大小
            iterations: 测试迭代次数
        
        Returns:
            dict: 性能结果
        """
        print(f"\n测试配置: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
        print("-" * 60)
        
        # 创建测试数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_size).to(self.device)
        
        # PyTorch原生LayerNorm
        native_ln = nn.LayerNorm(hidden_size).to(self.device)
        native_ln.eval()
        
        results = {
            'config': {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'hidden_size': hidden_size,
                'total_elements': batch_size * seq_len * hidden_size
            }
        }
        
        # 测试PyTorch原生实现
        print("测试PyTorch原生LayerNorm...")
        self.warmup(native_ln, input_tensor)
        
        latencies_native = []
        with torch.no_grad():
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                for _ in tqdm(range(iterations), desc="Native LayerNorm"):
                    start_event.record()
                    output = native_ln(input_tensor)
                    end_event.record()
                    torch.cuda.synchronize()
                    latencies_native.append(start_event.elapsed_time(end_event))
            else:
                for _ in tqdm(range(iterations), desc="Native LayerNorm"):
                    start = time.time()
                    output = native_ln(input_tensor)
                    end = time.time()
                    latencies_native.append((end - start) * 1000)
        
        latencies_native = np.array(latencies_native)
        results['native'] = {
            'mean_ms': float(np.mean(latencies_native)),
            'std_ms': float(np.std(latencies_native)),
            'min_ms': float(np.min(latencies_native)),
            'max_ms': float(np.max(latencies_native)),
            'p50_ms': float(np.percentile(latencies_native, 50)),
            'p95_ms': float(np.percentile(latencies_native, 95)),
            'p99_ms': float(np.percentile(latencies_native, 99)),
        }
        
        print(f"  平均延迟: {results['native']['mean_ms']:.4f} ± {results['native']['std_ms']:.4f} ms")
        
        # 测试自定义实现
        if CUDA_EXTENSION_AVAILABLE and torch.cuda.is_available():
            # 基础版本
            print("\n测试自定义LayerNorm (基础版本)...")
            custom_ln_basic = CustomLayerNorm(hidden_size, use_optimized=False).to(self.device)
            custom_ln_basic.weight.data = native_ln.weight.data.clone()
            custom_ln_basic.bias.data = native_ln.bias.data.clone()
            custom_ln_basic.eval()
            
            self.warmup(custom_ln_basic, input_tensor)
            
            latencies_basic = []
            with torch.no_grad():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                for _ in tqdm(range(iterations), desc="Custom LayerNorm (Basic)"):
                    start_event.record()
                    output = custom_ln_basic(input_tensor)
                    end_event.record()
                    torch.cuda.synchronize()
                    latencies_basic.append(start_event.elapsed_time(end_event))
            
            latencies_basic = np.array(latencies_basic)
            results['custom_basic'] = {
                'mean_ms': float(np.mean(latencies_basic)),
                'std_ms': float(np.std(latencies_basic)),
                'min_ms': float(np.min(latencies_basic)),
                'max_ms': float(np.max(latencies_basic)),
                'p50_ms': float(np.percentile(latencies_basic, 50)),
                'p95_ms': float(np.percentile(latencies_basic, 95)),
                'p99_ms': float(np.percentile(latencies_basic, 99)),
                'speedup': float(results['native']['mean_ms'] / np.mean(latencies_basic))
            }
            
            print(f"  平均延迟: {results['custom_basic']['mean_ms']:.4f} ± {results['custom_basic']['std_ms']:.4f} ms")
            print(f"  加速比: {results['custom_basic']['speedup']:.2f}x")
            
            # 优化版本（如果hidden_size可以被4整除）
            if hidden_size % 4 == 0:
                print("\n测试自定义LayerNorm (优化版本)...")
                custom_ln_opt = CustomLayerNorm(hidden_size, use_optimized=True).to(self.device)
                custom_ln_opt.weight.data = native_ln.weight.data.clone()
                custom_ln_opt.bias.data = native_ln.bias.data.clone()
                custom_ln_opt.eval()
                
                self.warmup(custom_ln_opt, input_tensor)
                
                latencies_opt = []
                with torch.no_grad():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    for _ in tqdm(range(iterations), desc="Custom LayerNorm (Optimized)"):
                        start_event.record()
                        output = custom_ln_opt(input_tensor)
                        end_event.record()
                        torch.cuda.synchronize()
                        latencies_opt.append(start_event.elapsed_time(end_event))
                
                latencies_opt = np.array(latencies_opt)
                results['custom_optimized'] = {
                    'mean_ms': float(np.mean(latencies_opt)),
                    'std_ms': float(np.std(latencies_opt)),
                    'min_ms': float(np.min(latencies_opt)),
                    'max_ms': float(np.max(latencies_opt)),
                    'p50_ms': float(np.percentile(latencies_opt, 50)),
                    'p95_ms': float(np.percentile(latencies_opt, 95)),
                    'p99_ms': float(np.percentile(latencies_opt, 99)),
                    'speedup': float(results['native']['mean_ms'] / np.mean(latencies_opt))
                }
                
                print(f"  平均延迟: {results['custom_optimized']['mean_ms']:.4f} ± {results['custom_optimized']['std_ms']:.4f} ms")
                print(f"  加速比: {results['custom_optimized']['speedup']:.2f}x")
        
        return results
    
    def benchmark_multiple_configs(self):
        """测试多个配置"""
        print("\n" + "="*80)
        print("多配置性能测试")
        print("="*80)
        
        # 测试配置列表
        configs = [
            # (batch_size, seq_len, hidden_size)
            (8, 64, 768),      # 小batch
            (16, 128, 768),    # 中等batch
            (32, 128, 768),    # BERT-base标准
            (64, 128, 768),    # 大batch
            (32, 256, 768),    # 长序列
            (32, 128, 1024),   # BERT-large
            (16, 512, 768),    # 更长序列
        ]
        
        all_results = []
        
        for batch_size, seq_len, hidden_size in configs:
            result = self.benchmark_single_config(
                batch_size, seq_len, hidden_size, 
                iterations=500
            )
            all_results.append(result)
        
        self.results['multiple_configs'] = all_results
        return all_results
    
    def compare_with_bert(self):
        """在实际BERT模型中对比性能"""
        if not CUDA_EXTENSION_AVAILABLE or not torch.cuda.is_available():
            print("\n跳过BERT对比测试（需要CUDA和自定义扩展）")
            return
        
        print("\n" + "="*80)
        print("BERT模型中LayerNorm性能对比")
        print("="*80)
        
        try:
            from transformers import BertModel
            from custom_layernorm import replace_layernorm_in_model
            
            # 加载BERT模型
            print("\n加载BERT模型...")
            model_native = BertModel.from_pretrained("bert-base-uncased").to(self.device)
            model_native.eval()
            
            # 创建带自定义LayerNorm的模型
            model_custom = BertModel.from_pretrained("bert-base-uncased").to(self.device)
            model_custom = replace_layernorm_in_model(model_custom, use_optimized=True)
            model_custom.eval()
            
            # 准备输入
            batch_size = 32
            seq_len = 128
            input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(self.device)
            attention_mask = torch.ones(batch_size, seq_len).to(self.device)
            
            iterations = 100
            
            # 测试原生模型
            print("\n测试原生BERT模型...")
            self.warmup(lambda x: model_native(input_ids=x, attention_mask=attention_mask), input_ids, 5)
            
            latencies_native = []
            with torch.no_grad():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                for _ in tqdm(range(iterations), desc="Native BERT"):
                    start_event.record()
                    outputs = model_native(input_ids=input_ids, attention_mask=attention_mask)
                    end_event.record()
                    torch.cuda.synchronize()
                    latencies_native.append(start_event.elapsed_time(end_event))
            
            # 测试自定义模型
            print("\n测试自定义LayerNorm的BERT模型...")
            self.warmup(lambda x: model_custom(input_ids=x, attention_mask=attention_mask), input_ids, 5)
            
            latencies_custom = []
            with torch.no_grad():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                for _ in tqdm(range(iterations), desc="Custom BERT"):
                    start_event.record()
                    outputs = model_custom(input_ids=input_ids, attention_mask=attention_mask)
                    end_event.record()
                    torch.cuda.synchronize()
                    latencies_custom.append(start_event.elapsed_time(end_event))
            
            # 统计结果
            latencies_native = np.array(latencies_native)
            latencies_custom = np.array(latencies_custom)
            
            print("\n性能对比结果:")
            print(f"  原生BERT平均延迟: {np.mean(latencies_native):.2f} ± {np.std(latencies_native):.2f} ms")
            print(f"  自定义LayerNorm BERT平均延迟: {np.mean(latencies_custom):.2f} ± {np.std(latencies_custom):.2f} ms")
            print(f"  加速比: {np.mean(latencies_native) / np.mean(latencies_custom):.3f}x")
            print(f"  性能提升: {(1 - np.mean(latencies_custom) / np.mean(latencies_native)) * 100:.2f}%")
            
            self.results['bert_comparison'] = {
                'native_mean_ms': float(np.mean(latencies_native)),
                'native_std_ms': float(np.std(latencies_native)),
                'custom_mean_ms': float(np.mean(latencies_custom)),
                'custom_std_ms': float(np.std(latencies_custom)),
                'speedup': float(np.mean(latencies_native) / np.mean(latencies_custom)),
                'improvement_percent': float((1 - np.mean(latencies_custom) / np.mean(latencies_native)) * 100)
            }
            
        except Exception as e:
            print(f"BERT对比测试失败: {e}")
    
    def plot_results(self, output_dir="./performance_plots"):
        """绘制性能对比图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        if 'multiple_configs' not in self.results:
            print("没有可绘制的结果")
            return
        
        results = self.results['multiple_configs']
        
        # 提取数据
        configs = []
        native_times = []
        custom_basic_times = []
        custom_opt_times = []
        speedups = []
        
        for result in results:
            config = result['config']
            config_str = f"B{config['batch_size']}\nS{config['seq_len']}\nH{config['hidden_size']}"
            configs.append(config_str)
            
            native_times.append(result['native']['mean_ms'])
            
            if 'custom_basic' in result:
                custom_basic_times.append(result['custom_basic']['mean_ms'])
            
            if 'custom_optimized' in result:
                custom_opt_times.append(result['custom_optimized']['mean_ms'])
                speedups.append(result['custom_optimized']['speedup'])
        
        # 绘制延迟对比图
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 子图1: 延迟对比
        x = np.arange(len(configs))
        width = 0.25
        
        axes[0].bar(x - width, native_times, width, label='Native', alpha=0.8)
        if custom_basic_times:
            axes[0].bar(x, custom_basic_times, width, label='Custom Basic', alpha=0.8)
        if custom_opt_times:
            axes[0].bar(x + width, custom_opt_times, width, label='Custom Optimized', alpha=0.8)
        
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('LayerNorm Performance Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(configs, fontsize=8)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # 子图2: 加速比
        if speedups:
            axes[1].bar(x, speedups, color='green', alpha=0.7)
            axes[1].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
            axes[1].set_xlabel('Configuration')
            axes[1].set_ylabel('Speedup')
            axes[1].set_title('Speedup over Native LayerNorm')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(configs, fontsize=8)
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'layernorm_performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n性能对比图已保存到: {plot_path}")
        plt.close()
    
    def save_results(self, output_path="./performance_comparison_results.json"):
        """保存结果到JSON文件"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n详细结果已保存到: {output_path}")
    
    def run_complete_comparison(self):
        """运行完整的性能对比测试"""
        print("\n" + "#"*80)
        print("# LayerNorm性能完整对比测试")
        print("#"*80)
        
        # 1. 多配置测试
        self.benchmark_multiple_configs()
        
        # 2. BERT模型对比
        self.compare_with_bert()
        
        # 3. 绘制结果
        self.plot_results()
        
        # 4. 保存结果
        self.save_results()
        
        print("\n" + "#"*80)
        print("# 性能对比测试完成!")
        print("#"*80)


def main():
    """主函数"""
    if not torch.cuda.is_available():
        print("错误: 该脚本需要CUDA支持")
        return
    
    if not CUDA_EXTENSION_AVAILABLE:
        print("警告: 自定义CUDA扩展不可用，只能测试PyTorch原生实现")
        print("请先编译CUDA扩展: python setup.py install")
        print("将继续运行基准测试...")
    
    # 创建对比工具
    comparison = LayerNormPerformanceComparison()
    
    # 运行完整对比
    comparison.run_complete_comparison()


if __name__ == "__main__":
    main()

