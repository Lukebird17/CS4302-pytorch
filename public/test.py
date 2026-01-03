import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.profiler import profile, ProfilerActivity, record_function
import os

# 使用环境变量或HF Mirror，不要硬编码token
# os.environ["HF_TOKEN"] = "your_token_here"  # 从环境变量获取
class BertSoftmaxResearch:
    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model_name = model_name

    def run_benchmark(self, dataset_name, num_labels, batch_sizes=[1,4, 8, 16, 32,64,128], seq_len=128):
        print(f"\n>>> 正在加载数据集: {dataset_name} (标签数: {num_labels})")
        dataset = load_dataset(dataset_name, split='test', use_auth_token=True)
        # 预先取样
        samples = dataset.shuffle(seed=42).select(range(max(batch_sizes)))
        
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels).to(self.device)
        model.eval()

        results = []
        for bs in batch_sizes:
            print(f"正在测试 Batch Size: {bs}...")
            batch_texts = samples['text'][:bs]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding='max_length', 
                                   truncation=True, max_length=seq_len).to(self.device)

            # GPU 预热
            with torch.no_grad():
                for _ in range(5):
                    model(**inputs)
            
            # 使用 PyTorch Profiler 进行性能分析
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False
            ) as prof:
                with torch.no_grad():
                    with record_function("bert_inference"):
                        model(**inputs)
            
            # 统计时间
            key_averages = prof.key_averages()
            total_inference_time = sum(e.cpu_time_total + (e.cuda_time_total if hasattr(e, 'cuda_time_total') else 0) 
                                       for e in key_averages if e.key == "bert_inference")
            
            softmax_events = [e for e in key_averages if "softmax" in e.key.lower()]
            softmax_abs_time = sum(e.cpu_time_total + (e.cuda_time_total if hasattr(e, 'cuda_time_total') else 0) 
                                   for e in softmax_events)
            
            results.append({
                "BatchSize": bs,
                "TotalTime_us": total_inference_time,
                "SoftmaxAbsTime_us": softmax_abs_time,
                "SoftmaxRelTime_%": (softmax_abs_time / total_inference_time * 100) if total_inference_time > 0 else 0
            })

        return pd.DataFrame(results)

# 执行调研、
researcher = BertSoftmaxResearch()
imdb_df = researcher.run_benchmark("imdb", num_labels=2)
ag_news_df = researcher.run_benchmark("ag_news", num_labels=4)


# 打印结果并保存
print(f"\n设备信息: {researcher.device_name}")
print("\nIMDB 调研结果:")
print(imdb_df)
print("\nAG News 调研结果:")
print(ag_news_df)

imdb_df.to_csv("imdb_softmax_results.csv", index=False)
ag_news_df.to_csv("ag_news_softmax_results.csv", index=False)