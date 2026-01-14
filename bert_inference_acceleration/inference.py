"""
BERT模型推理脚本 - IMDB情感分类
"""
import torch
import torch.nn as nn
import sys
import os
import time
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.optimized_bert import create_optimized_bert_model
from data.imdb_loader import get_dataloader


def inference(model, dataloader, device='cuda', verbose=True):
    """
    执行推理并计算准确率
    
    Args:
        model: BERT模型
        dataloader: 数据加载器
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        accuracy: 准确率
        avg_time: 平均推理时间(ms)
    """
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    times = []
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="推理中") if verbose else dataloader
        
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 预热
            if len(times) == 0:
                _ = model(input_ids, attention_mask, token_type_ids)
                if device == 'cuda':
                    torch.cuda.synchronize()
            
            # 计时
            start_time = time.perf_counter()
            
            logits = model(input_ids, attention_mask, token_type_ids)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            batch_time = (end_time - start_time) * 1000  # 转换为ms
            times.append(batch_time)
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if verbose and len(times) % 10 == 0:
                current_acc = correct / total
                iterator.set_postfix({
                    'acc': f'{current_acc:.4f}',
                    'avg_time': f'{sum(times)/len(times):.2f}ms'
                })
    
    accuracy = correct / total
    avg_time = sum(times) / len(times) if times else 0
    
    return accuracy, avg_time, times


def main():
    parser = argparse.ArgumentParser(description='BERT推理加速 - IMDB情感分类')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='预训练模型名称')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='推理样本数量 (0表示全部)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载workers数量')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("BERT推理加速项目 - IMDB情感分类")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大序列长度: {args.max_length}")
    print("=" * 60)
    
    # 加载数据
    print("\n正在加载数据...")
    dataloader = get_dataloader(
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length
    )
    
    # 限制样本数量
    if args.num_samples > 0:
        num_batches = min(args.num_samples // args.batch_size, len(dataloader))
        dataloader = list(dataloader)[:num_batches]
        print(f"使用 {num_batches * args.batch_size} 个样本进行测试")
    
    # 创建模型
    print("\n正在加载优化的BERT模型...")
    model = create_optimized_bert_model(args.model_name, num_labels=2)
    
    # 推理
    print("\n开始推理...")
    accuracy, avg_time, times = inference(model, dataloader, device=args.device)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("推理结果统计")
    print("=" * 60)
    print(f"准确率: {accuracy * 100:.2f}%")
    print(f"平均推理时间: {avg_time:.2f} ms/batch")
    print(f"吞吐量: {args.batch_size / (avg_time / 1000):.2f} samples/s")
    
    if len(times) > 1:
        times_sorted = sorted(times)
        p50 = times_sorted[len(times_sorted) // 2]
        p95 = times_sorted[int(len(times_sorted) * 0.95)]
        p99 = times_sorted[int(len(times_sorted) * 0.99)]
        print(f"延迟分布:")
        print(f"  P50: {p50:.2f} ms")
        print(f"  P95: {p95:.2f} ms")
        print(f"  P99: {p99:.2f} ms")
    
    print("=" * 60)


if __name__ == '__main__':
    main()




