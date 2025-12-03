#!/usr/bin/env python3
"""
BERT文本分类训练脚本
支持数据集: IMDB, AG News
用于CUDA算子profiling和性能测试
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from torch.profiler import profile, record_function, ProfilerActivity
import time


class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_imdb_dataset(tokenizer, max_length=128):
    """加载IMDB数据集"""
    print("加载IMDB数据集...")
    dataset = load_dataset("imdb")
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # 为了快速测试，可以取子集
    # train_texts = train_texts[:1000]
    # train_labels = train_labels[:1000]
    # test_texts = test_texts[:200]
    # test_labels = test_labels[:200]
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数: 2 (positive/negative)")
    
    return train_dataset, test_dataset, 2


def load_agnews_dataset(tokenizer, max_length=128):
    """加载AG News数据集"""
    print("加载AG News数据集...")
    dataset = load_dataset("ag_news")
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数: 4 (World/Sports/Business/Sci-Tech)")
    
    return train_dataset, test_dataset, 4


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def profile_inference(model, dataloader, device, output_dir='./profiling_results'):
    """Profiling推理性能"""
    print("\n" + "="*80)
    print("开始Profiling推理性能...")
    print("="*80)
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 预热
    print("预热GPU...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    
    # Profiling
    print("执行Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        with record_function("bert_inference"):
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 20:  # Profile 20个batch
                        break
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    with record_function(f"batch_{i}"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    
    # 打印结果
    print("\n" + "="*80)
    print("CUDA Kernel 热点分析 (按CUDA时间排序)")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_name_column_width=80
    ))
    
    # 保存trace文件
    trace_file = os.path.join(output_dir, 'bert_inference_trace.json')
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace已保存到: {trace_file}")
    
    # 提取关键算子统计
    key_ops = {}
    for evt in prof.key_averages():
        name = evt.key
        if any(keyword in name.lower() for keyword in [
            'softmax', 'layernorm', 'layer_norm', 'gemm', 'matmul', 'bmm',
            'addmm', 'gelu', 'attention', 'linear', 'mm'
        ]):
            key_ops[name] = {
                'cuda_time_total_ms': evt.cuda_time_total / 1000.0,
                'cpu_time_total_ms': evt.cpu_time_total / 1000.0,
                'count': evt.count,
                'cuda_time_avg_ms': evt.cuda_time / 1000.0 if evt.count > 0 else 0,
            }
    
    # 排序并保存
    sorted_ops = sorted(key_ops.items(), key=lambda x: x[1]['cuda_time_total_ms'], reverse=True)
    
    stats_file = os.path.join(output_dir, 'kernel_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'sorted_kernels': [{
                'name': name,
                **stats
            } for name, stats in sorted_ops],
            'summary': {
                'total_kernels': len(sorted_ops),
                'top_10_time_ms': sum(stats['cuda_time_total_ms'] for _, stats in sorted_ops[:10])
            }
        }, f, indent=2)
    
    print(f"Kernel统计已保存到: {stats_file}")
    
    print("\n" + "="*80)
    print("Top 20 关键算子:")
    print("="*80)
    print(f"{'算子名称':<60} {'总时间(ms)':<15} {'调用次数':<10} {'平均(ms)'}")
    print("-" * 100)
    for name, stats in sorted_ops[:20]:
        print(f"{name:<60} {stats['cuda_time_total_ms']:<15.3f} "
              f"{stats['count']:<10} {stats['cuda_time_avg_ms']:.3f}")


def benchmark_inference_speed(model, dataloader, device, num_batches=100):
    """Benchmark推理速度"""
    print("\n" + "="*80)
    print("Benchmark推理速度...")
    print("="*80)
    
    model.eval()
    
    # 预热
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    
    # Benchmark
    total_time = 0
    total_samples = 0
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        start_event.record()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)
            
            batch_start.record()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_end.record()
            
            torch.cuda.synchronize()
            
            total_samples += input_ids.size(0)
        
        end_event.record()
    
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    
    avg_time_per_batch = total_time / num_batches
    throughput = total_samples / total_time
    
    print(f"\n总样本数: {total_samples}")
    print(f"总时间: {total_time:.3f} 秒")
    print(f"平均每batch时间: {avg_time_per_batch*1000:.2f} ms")
    print(f"吞吐量: {throughput:.2f} samples/sec")
    
    return {
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'throughput': throughput,
        'total_samples': total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='BERT文本分类训练和Profiling')
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'agnews'],
                        help='数据集选择: imdb或agnews')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='预训练模型名称')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--do_train', action='store_true',
                        help='是否训练')
    parser.add_argument('--do_eval', action='store_true',
                        help='是否评估')
    parser.add_argument('--do_profile', action='store_true',
                        help='是否进行profiling')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='加载checkpoint路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}\n")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载tokenizer
    print(f"加载tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # 加载数据集
    if args.dataset == 'imdb':
        train_dataset, test_dataset, num_labels = load_imdb_dataset(tokenizer, args.max_length)
    else:
        train_dataset, test_dataset, num_labels = load_agnews_dataset(tokenizer, args.max_length)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 加载模型
    if args.load_checkpoint:
        print(f"从checkpoint加载模型: {args.load_checkpoint}")
        model = BertForSequenceClassification.from_pretrained(args.load_checkpoint)
    else:
        print(f"加载预训练模型: {args.model_name}")
        model = BertForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels
        )
    
    model.to(device)
    
    # 训练
    if args.do_train:
        print("\n" + "="*80)
        print("开始训练")
        print("="*80)
        
        # 优化器和调度器
        optimizer = AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        best_acc = 0
        training_stats = []
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            print(f"\nEpoch {epoch}:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  训练精度: {train_acc:.4f}")
            
            # 评估
            eval_loss, eval_acc = evaluate(model, test_loader, device)
            print(f"  验证损失: {eval_loss:.4f}")
            print(f"  验证精度: {eval_acc:.4f}")
            
            training_stats.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc
            })
            
            # 保存最佳模型
            if eval_acc > best_acc:
                best_acc = eval_acc
                save_path = os.path.join(args.output_dir, 'best_model')
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  保存最佳模型到: {save_path}")
        
        # 保存训练统计
        stats_file = os.path.join(args.output_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(training_stats, f, indent=2)
        print(f"\n训练统计已保存到: {stats_file}")
        print(f"最佳验证精度: {best_acc:.4f}")
    
    # 评估
    if args.do_eval:
        print("\n" + "="*80)
        print("评估模型")
        print("="*80)
        eval_loss, eval_acc = evaluate(model, test_loader, device)
        print(f"测试损失: {eval_loss:.4f}")
        print(f"测试精度: {eval_acc:.4f}")
    
    # Profiling
    if args.do_profile:
        profile_output_dir = os.path.join(args.output_dir, 'profiling_results')
        profile_inference(model, test_loader, device, profile_output_dir)
        
        # Benchmark速度
        benchmark_stats = benchmark_inference_speed(model, test_loader, device)
        
        # 保存benchmark结果
        benchmark_file = os.path.join(profile_output_dir, 'benchmark_stats.json')
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_stats, f, indent=2)
        print(f"\nBenchmark结果已保存到: {benchmark_file}")


if __name__ == '__main__':
    main()










