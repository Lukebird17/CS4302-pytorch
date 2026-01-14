import os
import torch
import pandas as pd
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_from_disk
from torch.profiler import profile, ProfilerActivity, record_function

# 配置基础路径
# 从环境变量读取 HF Token（如果需要）
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# 自动检测项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # lhl/
BASE_DIR = PROJECT_ROOT
MODEL_DIR = os.path.join(BASE_DIR, "bert_inference_acceleration", "models")
DATASET_DIR = os.path.join(BASE_DIR, "bert_inference_acceleration", "dataset")

# 新增：定义统一的输出根目录 /hy-tmp/fj/output
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")

# 算子关键字映射 (保持 copy 关键字以捕捉 transpose 物理搬运)
OP_KEYWORDS = {
    "softmax": ["softmax"],
    "layernorm": ["layer_norm", "layernorm", "native_layer_norm"],
    "addmm": ["addmm", "gemm", "mm"],
    "transpose": ["transpose", "permute", "contiguous", "copy"] 
}

class BertOperatorResearch:
    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        local_model_path = os.path.join(MODEL_DIR, model_name.replace("/", "_"))
        self.tokenizer = BertTokenizer.from_pretrained(local_model_path)
        self.model = BertForSequenceClassification.from_pretrained(local_model_path).to(self.device)

    def run_benchmark(self, op_type, dataset_name, num_labels, batch_sizes=[1, 4, 8, 16, 32, 64, 128], seq_len=128):
        print(f"\n" + "="*50)
        print(f">>> 任务: {op_type.upper()} | 数据集: {dataset_name}")
        print("="*50)
        
        # 修改点：结果保存路径重构为 ./output/{op_type}
        save_dir = os.path.join(OUTPUT_ROOT, op_type)
        os.makedirs(save_dir, exist_ok=True)

        local_data_path = os.path.join(DATASET_DIR, dataset_name.replace("/", "_"))
        dataset = load_from_disk(local_data_path)
        test_data = dataset['test'] if 'test' in dataset else dataset['train']
        samples = test_data.shuffle(seed=42).select(range(max(batch_sizes)))
        
        if self.model.num_labels != num_labels:
            self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels).to(self.device)
            self.model.num_labels = num_labels

        self.model.eval()
        results = []
        target_kws = OP_KEYWORDS.get(op_type, [op_type])
        
        for bs in batch_sizes:
            print(f"Profiling Batch Size: {bs}...")
            batch_texts = samples['text'][:bs]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding='max_length', 
                                   truncation=True, max_length=seq_len).to(self.device)

            torch.cuda.empty_cache()
            with torch.no_grad():
                for _ in range(5): self.model(**inputs)
            torch.cuda.synchronize()
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False
            ) as prof:
                with torch.no_grad():
                    with record_function("bert_inference"):
                        self.model(**inputs)
                        torch.cuda.synchronize() 
            
            key_averages = prof.key_averages()
            
            # 统一 CUDA 时间口径
            total_cuda_time = sum(e.cuda_time_total for e in key_averages)
            op_events = [e for e in key_averages if any(kw in e.key.lower() for kw in target_kws)]
            abs_time = sum(e.cuda_time_total for e in op_events)
            
            kernels = set()
            for e in key_averages:
                if e.device_type == torch.profiler.DeviceType.CUDA and e.cuda_time_total > 0:
                    if any(kw in e.key.lower() for kw in target_kws):
                        kernels.add(e.key)

            results.append({
                "BatchSize": bs,
                "TotalTime_us": total_cuda_time,
                "AbsTime_us": abs_time,
                "RelTime_%": (abs_time / total_cuda_time * 100) if total_cuda_time > 0 else 0,
                "CUDA_Kernels": " | ".join(kernels) if kernels else "N/A"
            })

        df = pd.DataFrame(results)
        output_file = os.path.join(save_dir, f"{dataset_name}_{op_type}_final.csv")
        df.to_csv(output_file, index=False)
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Operator Profiler")
    parser.add_argument("--op", type=str, default="softmax", 
                        choices=["softmax", "layernorm", "addmm", "transpose"],
                        help="选择要测试的算子")
    args = parser.parse_args()

    researcher = BertOperatorResearch("bert-base-uncased")
    researcher.run_benchmark(args.op, "imdb", num_labels=2)
    researcher.run_benchmark(args.op, "ag_news", num_labels=4)