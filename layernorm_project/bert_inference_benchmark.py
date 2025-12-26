"""
BERT推理性能评测代码
使用bert-base-uncased模型在IMDB数据集上进行推理性能测试
用于评测LayerNorm算子的实际性能影响
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except Exception as e:
    DATASETS_AVAILABLE = False
    print(f"警告: datasets库不可用 ({e})")
    print("将使用模拟数据替代IMDB数据集")
import time
from torch.profiler import profile, ProfilerActivity
import numpy as np
from tqdm import tqdm
import json
import os


class BERTInferenceBenchmark:
    """BERT推理性能评测工具"""
    
    def __init__(self, model_name="bert-base-uncased", max_length=128):
        """
        初始化评测工具
        Args:
            model_name: BERT模型名称
            max_length: 最大序列长度
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，将使用CPU进行评测")
        
        self.model_name = model_name
        self.max_length = max_length
        
        # 加载tokenizer和模型
        print(f"\n正在加载模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"模型加载完成")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        # 统计LayerNorm层数量
        self.count_layernorm_layers()
        
    def count_layernorm_layers(self):
        """统计模型中LayerNorm层的数量"""
        layernorm_count = 0
        layernorm_info = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                layernorm_count += 1
                normalized_shape = module.normalized_shape
                layernorm_info.append((name, normalized_shape))
        
        print(f"\n模型中LayerNorm层数量: {layernorm_count}")
        print("LayerNorm层详细信息:")
        for name, shape in layernorm_info[:5]:  # 只显示前5个
            print(f"  {name}: normalized_shape={shape}")
        if len(layernorm_info) > 5:
            print(f"  ... 还有 {len(layernorm_info) - 5} 个LayerNorm层")
        
        self.layernorm_count = layernorm_count
        self.layernorm_info = layernorm_info
        
    def load_imdb_dataset(self, num_samples=100):
        """
        加载IMDB数据集
        Args:
            num_samples: 加载的样本数量
        Returns:
            样本列表
        """
        print(f"\n正在加载IMDB数据集 (前{num_samples}个样本)...")
        
        if not DATASETS_AVAILABLE:
            print("datasets库不可用，使用模拟数据...")
            return self._generate_dummy_samples(num_samples)
        
        try:
            # 加载IMDB测试集
            dataset = load_dataset("imdb", split=f"test[:{num_samples}]")
            samples = [item['text'] for item in dataset]
            print(f"成功加载 {len(samples)} 个样本")
            return samples
        except Exception as e:
            print(f"加载数据集失败: {e}")
            print("使用模拟数据替代...")
            return self._generate_dummy_samples(num_samples)
    
    def _generate_dummy_samples(self, num_samples):
        """生成模拟数据"""
        base_samples = [
            "This movie is absolutely fantastic! I loved every minute of it. The acting was superb and the plot kept me engaged throughout.",
            "Terrible film, complete waste of time and money. The story made no sense and the characters were poorly developed.",
            "An interesting plot with great character development. The cinematography was beautiful and the soundtrack was memorable.",
            "I had mixed feelings about this movie. Some parts were excellent while others felt rushed and incomplete.",
            "One of the best films I've seen this year. Highly recommended for anyone who enjoys thoughtful drama.",
            "Boring and predictable. I found myself checking my watch multiple times during the screening.",
            "A masterpiece of modern cinema. The director's vision was clear and executed perfectly.",
            "Not bad, but not great either. It's an okay way to spend an evening if you have nothing else to do.",
        ]
        samples = []
        for i in range(num_samples):
            samples.append(base_samples[i % len(base_samples)])
        print(f"生成了 {len(samples)} 个模拟样本")
        return samples
    
    def prepare_batch(self, texts, batch_size=None):
        """
        准备批次数据
        Args:
            texts: 文本列表
            batch_size: 批次大小
        Returns:
            tokenized inputs
        """
        if batch_size is None:
            batch_size = len(texts)
        
        # Tokenization
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def warmup(self, num_iterations=10):
        """预热模型"""
        print(f"\n正在预热模型 ({num_iterations}次迭代)...")
        
        dummy_text = ["This is a warmup text."] * 8
        inputs = self.prepare_batch(dummy_text)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(**inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        print("预热完成")
    
    def benchmark_inference(self, samples, batch_size=32, num_iterations=100):
        """
        推理性能基准测试
        Args:
            samples: 测试样本
            batch_size: 批次大小
            num_iterations: 迭代次数
        """
        print("\n" + "="*80)
        print("BERT推理性能测试")
        print("="*80)
        
        # 预热
        self.warmup()
        
        # 准备数据
        test_samples = samples[:batch_size]
        inputs = self.prepare_batch(test_samples, batch_size)
        
        print(f"\n测试配置:")
        print(f"  批次大小: {batch_size}")
        print(f"  序列长度: {self.max_length}")
        print(f"  迭代次数: {num_iterations}")
        
        # 开始测试
        latencies = []
        
        with torch.no_grad():
            if torch.cuda.is_available():
                # 使用CUDA Event计时
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 预热一次
                _ = self.model(**inputs)
                torch.cuda.synchronize()
                
                # 正式测试
                for _ in tqdm(range(num_iterations), desc="推理中"):
                    start_event.record()
                    outputs = self.model(**inputs)
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    latency = start_event.elapsed_time(end_event)
                    latencies.append(latency)
            else:
                # CPU计时
                for _ in tqdm(range(num_iterations), desc="推理中"):
                    start_time = time.time()
                    outputs = self.model(**inputs)
                    end_time = time.time()
                    latencies.append((end_time - start_time) * 1000)
        
        # 统计结果
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # 计算吞吐量
        throughput = (batch_size * num_iterations) / (np.sum(latencies) / 1000)  # samples/sec
        tokens_per_sec = throughput * self.max_length
        
        print("\n性能统计:")
        print(f"  平均延迟: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  最小延迟: {min_latency:.2f} ms")
        print(f"  最大延迟: {max_latency:.2f} ms")
        print(f"  P50 延迟: {p50_latency:.2f} ms")
        print(f"  P95 延迟: {p95_latency:.2f} ms")
        print(f"  P99 延迟: {p99_latency:.2f} ms")
        print(f"  吞吐量: {throughput:.2f} samples/sec")
        print(f"  Token吞吐量: {tokens_per_sec:.2f} tokens/sec")
        
        results = {
            'batch_size': batch_size,
            'sequence_length': self.max_length,
            'avg_latency_ms': float(avg_latency),
            'std_latency_ms': float(std_latency),
            'min_latency_ms': float(min_latency),
            'max_latency_ms': float(max_latency),
            'p50_latency_ms': float(p50_latency),
            'p95_latency_ms': float(p95_latency),
            'p99_latency_ms': float(p99_latency),
            'throughput_samples_per_sec': float(throughput),
            'throughput_tokens_per_sec': float(tokens_per_sec),
        }
        
        return results
    
    def profile_layernorm(self, samples, batch_size=8):
        """
        使用Profiler专门分析LayerNorm的性能
        """
        if not torch.cuda.is_available():
            print("跳过Profiler分析（需要CUDA）")
            return
            
        print("\n" + "="*80)
        print("LayerNorm性能Profiling")
        print("="*80)
        
        # 准备数据
        test_samples = samples[:batch_size]
        inputs = self.prepare_batch(test_samples, batch_size)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(**inputs)
            torch.cuda.synchronize()
        
        # Profiling
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            ) as prof:
                for _ in range(20):
                    outputs = self.model(**inputs)
                    torch.cuda.synchronize()
        
        # 分析结果
        print("\n--- LayerNorm相关操作 (按CUDA时间排序) ---")
        
        events = prof.key_averages()
        layernorm_events = [
            event for event in events 
            if 'layer_norm' in event.key.lower() or 'layernorm' in event.key.lower()
        ]
        
        total_cuda_time = sum(event.cuda_time_total for event in events if event.device_type == torch.profiler.DeviceType.CUDA)
        layernorm_cuda_time = sum(event.cuda_time_total for event in layernorm_events)
        
        if layernorm_events:
            print(f"{'操作名称':<50} {'调用次数':<10} {'总时间(us)':<15} {'平均时间(us)':<15}")
            print("-" * 90)
            
            for event in sorted(layernorm_events, key=lambda x: x.cuda_time_total, reverse=True):
                print(f"{event.key[:50]:<50} {event.count:<10} {event.cuda_time_total:<15.2f} {event.cuda_time:<15.2f}")
            
            print(f"\nLayerNorm占总CUDA时间的比例: {layernorm_cuda_time / total_cuda_time * 100:.2f}%")
        else:
            print("未找到LayerNorm相关操作（可能被融合或优化）")
        
        # 保存详细trace
        output_dir = "./profiler_results"
        os.makedirs(output_dir, exist_ok=True)
        trace_path = os.path.join(output_dir, "bert_layernorm_trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"\n已保存Chrome trace到: {trace_path}")
        
        return layernorm_cuda_time / total_cuda_time if total_cuda_time > 0 else 0
    
    def compare_batch_sizes(self, samples, batch_sizes=[1, 2, 4, 8, 16, 32]):
        """
        比较不同批次大小的性能
        """
        print("\n" + "="*80)
        print("不同批次大小性能对比")
        print("="*80)
        
        results = []
        
        print(f"\n{'批次大小':<12} {'平均延迟(ms)':<15} {'吞吐量(samples/s)':<20} {'每样本延迟(ms)':<18}")
        print("-" * 65)
        
        for batch_size in batch_sizes:
            if batch_size > len(samples):
                continue
                
            result = self.benchmark_inference(
                samples, 
                batch_size=batch_size, 
                num_iterations=50
            )
            
            per_sample_latency = result['avg_latency_ms'] / batch_size
            
            print(f"{batch_size:<12} {result['avg_latency_ms']:<15.2f} {result['throughput_samples_per_sec']:<20.2f} {per_sample_latency:<18.2f}")
            
            results.append({
                'batch_size': batch_size,
                **result
            })
        
        return results
    
    def save_benchmark_results(self, results, output_path="./benchmark_results.json"):
        """保存评测结果"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n评测结果已保存到: {output_path}")
    
    def run_complete_benchmark(self):
        """运行完整的评测流程"""
        print("\n" + "#"*80)
        print("# BERT模型LayerNorm性能完整评测")
        print("#"*80)
        
        if torch.cuda.is_available():
            print(f"\nGPU信息:")
            print(f"  设备名称: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # 加载数据集
        samples = self.load_imdb_dataset(num_samples=200)
        
        # 1. 基准测试
        print("\n【1. 基准性能测试】")
        base_results = self.benchmark_inference(samples, batch_size=32, num_iterations=100)
        
        # 2. LayerNorm profiling
        print("\n【2. LayerNorm Profiling分析】")
        layernorm_ratio = self.profile_layernorm(samples, batch_size=8)
        
        # 3. 不同批次大小对比
        print("\n【3. 批次大小对比测试】")
        batch_results = self.compare_batch_sizes(samples)
        
        # 保存结果
        all_results = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'layernorm_count': self.layernorm_count,
            'layernorm_ratio': float(layernorm_ratio) if layernorm_ratio else 0,
            'base_results': base_results,
            'batch_size_comparison': batch_results,
        }
        
        self.save_benchmark_results(all_results)
        
        print("\n" + "#"*80)
        print("# 评测完成!")
        print("#"*80)
        
        print("\n关键发现:")
        print(f"  1. 模型中共有 {self.layernorm_count} 个LayerNorm层")
        if layernorm_ratio:
            print(f"  2. LayerNorm占总CUDA时间的 {layernorm_ratio*100:.2f}%")
        print(f"  3. 基准性能: {base_results['avg_latency_ms']:.2f}ms (batch_size=32)")
        print(f"  4. 吞吐量: {base_results['throughput_samples_per_sec']:.2f} samples/sec")
        print("\n优化建议:")
        print("  - LayerNorm是Transformer的关键算子，优化空间显著")
        print("  - 可以考虑算子融合、混合精度、kernel优化等方法")
        print("  - 建议重点优化高频调用的LayerNorm kernel")


def main():
    """主函数"""
    print("BERT推理性能评测工具")
    print("="*80)
    
    # 创建评测工具
    benchmark = BERTInferenceBenchmark(
        model_name="bert-base-uncased",
        max_length=128
    )
    
    # 运行完整评测
    benchmark.run_complete_benchmark()


if __name__ == "__main__":
    main()

