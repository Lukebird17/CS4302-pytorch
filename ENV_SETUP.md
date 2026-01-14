# 环境变量配置说明

## Hugging Face Token 配置

### 为什么需要 Token？

只有在以下情况需要配置 Hugging Face Token：
- ❌ **不需要**：如果数据集和模型已经下载到本地
- ✅ **需要**：如果需要从 Hugging Face Hub 下载模型或数据集

### 配置方法

#### 方法 1：临时设置（推荐用于测试）

```bash
export HF_TOKEN="your_token_here"
python operator_search/test_new.py --op addmm
```

#### 方法 2：.env 文件（推荐用于开发）

1. 复制示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，添加你的 token：
```bash
HF_TOKEN=your_actual_token_here
```

3. 使用 python-dotenv 加载（可选）：
```bash
pip install python-dotenv
```

在脚本开头添加：
```python
from dotenv import load_dotenv
load_dotenv()
```

#### 方法 3：系统环境变量（推荐用于生产）

在 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
export HF_TOKEN="your_token_here"
```

然后重新加载：
```bash
source ~/.bashrc
```

### 获取 Hugging Face Token

1. 登录 https://huggingface.co/
2. 访问 Settings → Access Tokens
3. 创建新 Token（选择 Read 权限即可）
4. 复制 Token

### 安全提示

⚠️ **重要：永远不要将 Token 提交到 Git！**

已在 `.gitignore` 中排除：
- `.env`
- `*.key`
- `*.token`
- `secrets/`

### 本项目使用说明

由于数据集已经在本地（`bert_inference_acceleration/dataset/`），通常**不需要**配置 Token。

只有在以下情况需要：
- 重新下载数据集
- 使用其他 Hugging Face 模型或数据集

---

## 其他环境变量

### CUDA_VISIBLE_DEVICES

限制使用的 GPU：
```bash
export CUDA_VISIBLE_DEVICES=0  # 只使用 GPU 0
```

### LD_LIBRARY_PATH

CUDA 算子需要的库路径（通常由 `install.sh` 自动设置）：
```bash
export LD_LIBRARY_PATH=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH
```

---

## 快速检查

验证环境变量是否设置正确：

```bash
python << EOF
import os
print(f"HF_TOKEN: {'已设置' if os.environ.get('HF_TOKEN') else '未设置（本项目不需要）'}")
print(f"CUDA可用: {__import__('torch').cuda.is_available()}")
EOF
```
