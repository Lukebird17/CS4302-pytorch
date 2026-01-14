"""
IMDB数据集加载和预处理
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
import numpy as np


class IMDBDataset(Dataset):
    """IMDB数据集封装"""
    def __init__(self, split='test', tokenizer_name='bert-base-uncased', max_length=512):
        print(f"正在加载IMDB数据集 ({split} split)...")
        self.dataset = load_dataset('imdb', split=split)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        print(f"加载完成，共 {len(self.dataset)} 条数据")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_dataloader(split='test', batch_size=32, num_workers=4, max_length=512):
    """创建数据加载器"""
    dataset = IMDBDataset(split=split, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == '__main__':
    # 测试数据加载
    print("测试数据加载...")
    dataloader = get_dataloader(split='test', batch_size=8)
    
    for batch in dataloader:
        print(f"Batch shape:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  token_type_ids: {batch['token_type_ids'].shape}")
        print(f"  label: {batch['label'].shape}")
        break
    
    print("数据加载测试成功！")




