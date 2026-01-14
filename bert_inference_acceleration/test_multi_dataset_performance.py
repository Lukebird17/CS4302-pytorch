"""
BERT推理加速 - 多数据集算子性能评测
更新：适配 Hugging Face Arrow 格式数据集读取 (save_to_disk 格式)
"""

import torch
import torch.nn as nn
import time
import numpy as np
import os
from transformers import BertTokenizer
try:
    from datasets import load_from_disk
except ImportError:
    print("❌ 缺少 datasets 库，请运行: pip install datasets")
    exit(1)

# ================= 配置与环境 =================
DATASET_BASE_PATH = "/hy-tmp/lhl/bert_inference_acceleration/dataset"
TOKENIZER_PATH = "bert-base-uncased" 

os.environ['LD_LIBRARY_PATH'] = os.path.join(os.path.dirname(torch.__file__), 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    from custom_ops_cuda import (
        gemm_bias_add_layernorm,
        gemm_bias_gelu_add_layernorm
    )
    print("✅ 成功加载自定义算子库")
except ImportError:
    print("❌ 未能加载 custom_ops_cuda，请检查编译情况")
    exit(1)

# ================= 核心工具函数 =================

def get_real_avg_seq_len(dataset_name, sample_size=1000):
    """
    针对 load_from_disk 格式的目录读取并计算平均长度
    """
    # 匹配图片中的目录：AG News -> ag_news, IMDB -> imdb
    dir_name = dataset_name.lower().replace(" ", "_")
    path = os.path.join(DATASET_BASE_PATH, dir_name)
    
    print(f"🔍 正在从本地 Arrow 目录加载: {path}")
    
    try:
        # 1. 加载本地数据集
        data = load_from_disk(path)
        
        # 2. 处理 DatasetDict (包含 train/test 的情况)
        if isinstance(data, dict) or hasattr(data, 'keys'):
            # 优先选择 train，否则取第一个 split
            split = 'train' if 'train' in data else list(data.keys())[0]
            ds = data[split]
        else:
            ds = data
            
        # 3. 采样并获取文本列
        # 自动识别列名：通常为 'text' 或 'description'
        cols = ds.column_names
        text_col = 'text' if 'text' in cols else ('description' if 'description' in cols else cols[0])
        
        sample_ds = ds.select(range(min(len(ds), sample_size)))
        texts = sample_ds[text_col]
        
        # 4. 计算长度
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
        lengths = [len(tokenizer.encode(t, add_special_tokens=True, max_length=512, truncation=True)) for t in texts]
        
        avg_len = int(np.mean(lengths))
        print(f"📊 {dataset_name} 统计完成: 实际平均长度 = {avg_len}")
        return avg_len

    except Exception as e:
        default = 512 if "imdb" in dir_name else 128
        print(f"⚠️ 读取失败 ({e})，使用预设默认值: {default}")
        return default

# ================= 性能评测函数 =================

def simulate_bert_attention_output(batch_size, seq_len, hidden_size, num_runs=100):
    input_flat = torch.randn(batch_size * seq_len, hidden_size).cuda()
    weight = torch.randn(hidden_size, hidden_size).cuda()
    bias = torch.randn(hidden_size).cuda()
    residual = torch.randn(batch_size * seq_len, hidden_size).cuda()
    gamma = torch.ones(hidden_size).cuda()
    beta = torch.zeros(hidden_size).cuda()
    
    for _ in range(10): _ = torch.nn.functional.linear(input_flat, weight, bias)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        x = torch.nn.functional.linear(input_flat, weight, bias)
        x = x + residual
        x = torch.nn.functional.layer_norm(x, (hidden_size,), gamma, beta, 1e-12)
    torch.cuda.synchronize()
    py_time = (time.perf_counter() - t0) * 1000 / num_runs
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(num_runs):
        x = gemm_bias_add_layernorm(input_flat, weight.t().contiguous(), bias, residual, gamma, beta, 1e-12)
    torch.cuda.synchronize()
    cu_time = (time.perf_counter() - t1) * 1000 / num_runs
    return py_time, cu_time

def simulate_bert_ffn(batch_size, seq_len, hidden_size, intermediate_size, num_runs=100):
    input_flat = torch.randn(batch_size * seq_len, intermediate_size).cuda()
    weight = torch.randn(hidden_size, intermediate_size).cuda()
    bias = torch.randn(hidden_size).cuda()
    residual = torch.randn(batch_size * seq_len, hidden_size).cuda()
    gamma = torch.ones(hidden_size).cuda()
    beta = torch.zeros(hidden_size).cuda()

    for _ in range(10): _ = torch.nn.functional.linear(input_flat, weight, bias)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        x = torch.nn.functional.linear(input_flat, weight, bias)
        x = x + residual
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.layer_norm(x, (hidden_size,), gamma, beta, 1e-12)
    torch.cuda.synchronize()
    py_time = (time.perf_counter() - t0) * 1000 / num_runs

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(num_runs):
        x = gemm_bias_gelu_add_layernorm(input_flat, weight.t().contiguous(), bias, residual, gamma, beta, 1e-12)
    torch.cuda.synchronize()
    cu_time = (time.perf_counter() - t1) * 1000 / num_runs
    return py_time, cu_time

# ================= 主程序 =================

def main():
    print("="*85)
    print("BERT 推理加速算子评测 - Arrow 格式适配版")
    print("="*85)

    configs = [
        {"name": "IMDB", "batch_size": 16},
        {"name": "AG News", "batch_size": 32}
    ]
    
    results = []

    for cfg in configs:
        # 获取真实长度
        raw_len = get_real_avg_seq_len(cfg['name'])
        final_len = min(raw_len, 512)
        
        # 运行评测
        att_py, att_cu = simulate_bert_attention_output(cfg['batch_size'], final_len, 768)
        ffn_py, ffn_cu = simulate_bert_ffn(cfg['batch_size'], final_len, 768, 3072)
        
        results.append({"ds": cfg['name'], "len": final_len, "type": "Attn-Out", "py": att_py, "cu": att_cu})
        results.append({"ds": cfg['name'], "len": final_len, "type": "FFN-Layer", "py": ffn_py, "cu": ffn_cu})

    # 打印报表
    print("\n📊 性能总结报告")
    print("="*85)
    print(f"{'数据集':<12} {'场景':<12} {'平均长度':<10} {'PyTorch(ms)':<15} {'自定义算子(ms)':<15} {'加速比':<10}")
    print("-" * 85)
    for r in results:
        speedup = r['py'] / r['cu']
        print(f"{r['ds']:<12} {r['type']:<12} {r['len']:<10} {r['py']:>10.3f}      {r['cu']:>10.3f}       {speedup:>6.2f}x")
    print("="*85)

if __name__ == '__main__':
    main()