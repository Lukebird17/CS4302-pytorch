# Git 推送问题修复指南

## 问题总结

推送时遇到两个问题：
1. ❌ **密钥泄露**：`operator_search/test_new.py` 中硬编码了 Hugging Face Token
2. ⚠️ **大文件警告**：`dataset/imdb/unsupervised/data-00000-of-00001.arrow` (64MB)

## 已完成的修复

✅ **密钥问题已修复**：
- 移除了硬编码的 Token
- 改为从环境变量读取
- 创建了 `.gitignore` 排除敏感文件

✅ **配置文件已创建**：
- `.gitignore` - 排除不需要提交的文件
- `.env.example` - 环境变量示例
- `ENV_SETUP.md` - 详细配置说明

---

## 🔧 修复步骤

### 步骤 1：验证修复

```bash
# 查看修改后的文件
cat operator_search/test_new.py | grep -A 5 "配置基础路径"

# 确认没有硬编码的 token
grep -r "hf_" . --exclude-dir=.git
```

### 步骤 2：添加修改到暂存区

```bash
# 添加修复后的文件
git add operator_search/test_new.py
git add .gitignore
git add .env.example
git add ENV_SETUP.md
git add GIT_FIX_GUIDE.md
```

### 步骤 3：修改上一次提交（推荐）

```bash
# 方式 A：修改最后一次提交（如果还没推送成功）
git commit --amend --no-edit

# 或者，如果要修改提交信息
git commit --amend -m "fix: 移除硬编码的 HF Token，改用环境变量"
```

### 步骤 4：重新推送

```bash
# 如果是第一次推送
git push -u origin main

# 如果之前推送失败了，重试即可
git push
```

---

## 🗂️ 处理大文件问题（可选）

如果想移除数据集文件（减小仓库大小）：

### 方案 1：移除数据集（推荐）

数据集文件很大且不需要提交到 Git：

```bash
# 1. 从 Git 跟踪中移除（但保留本地文件）
git rm --cached -r bert_inference_acceleration/dataset/
git rm --cached -r operator_search/output/

# 2. 提交
git commit -m "chore: 从版本控制中移除大文件（dataset）"

# 3. 推送
git push
```

**注意**：`.gitignore` 已配置，之后不会再追踪这些文件。

### 方案 2：使用 Git LFS（如果需要版本控制大文件）

```bash
# 1. 安装 Git LFS
git lfs install

# 2. 追踪 .arrow 文件
git lfs track "*.arrow"

# 3. 添加 .gitattributes
git add .gitattributes

# 4. 提交并推送
git commit -m "chore: 使用 Git LFS 管理大文件"
git push
```

---

## 🚨 如果之前已经推送成功但有 Token

如果 Token 已经在历史提交中，需要清理 Git 历史：

### ⚠️ 警告：这会改写历史，需谨慎操作！

```bash
# 方式 1：使用 git filter-repo（推荐）
# 安装：pip install git-filter-repo

# 移除文件中的敏感信息
git filter-repo --path operator_search/test_new.py --invert-paths

# 或者使用 BFG Repo-Cleaner
# https://rtyley.github.io/bfg-repo-cleaner/

# 强制推送（会覆盖远程历史）
git push --force origin main
```

### ⚠️ 更安全的方式：撤销 Token

如果 Token 已经泄露到 GitHub：

1. **立即撤销 Token**：
   - 登录 https://huggingface.co/settings/tokens
   - 删除泄露的 Token
   - 创建新 Token

2. **清理 Git 历史**（参考上面的方法）

3. **重新推送**

---

## ✅ 验证修复

### 1. 本地验证

```bash
# 检查是否还有硬编码的 token
grep -r "hf_" . --exclude-dir=.git --exclude-dir=dataset

# 查看将要推送的文件
git diff origin/main
```

### 2. GitHub 验证

推送成功后，检查：
- ✅ Secret Scanning 警告消失
- ✅ 代码中没有硬编码的 Token
- ✅ `.gitignore` 正确配置

---

## 📝 提交信息建议

```bash
# 修复密钥泄露
git commit -m "fix: 移除硬编码的 HF Token，改用环境变量

- 从环境变量读取 HF_TOKEN
- 添加 .gitignore 排除敏感文件
- 创建 ENV_SETUP.md 说明配置方法
- 自动检测项目路径，不依赖硬编码路径"

# 移除大文件
git commit -m "chore: 从版本控制中移除数据集文件

- 数据集文件过大（>50MB）
- 添加到 .gitignore
- 用户需要自行下载或使用本地数据集"
```

---

## 🎯 快速修复命令（一键执行）

```bash
#!/bin/bash
echo "🔧 开始修复 Git 推送问题..."

# 1. 添加修复后的文件
git add operator_search/test_new.py .gitignore ENV_SETUP.md GIT_FIX_GUIDE.md

# 2. 修改最后一次提交
git commit --amend -m "fix: 移除硬编码的 HF Token，改用环境变量"

# 3. 尝试推送
if git push; then
    echo "✅ 推送成功！"
else
    echo "❌ 推送失败，请检查错误信息"
fi

# 4. 可选：移除大文件
read -p "是否要移除数据集文件？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git rm --cached -r bert_inference_acceleration/dataset/ 2>/dev/null || true
    git rm --cached -r operator_search/output/ 2>/dev/null || true
    git commit -m "chore: 从版本控制中移除大文件"
    git push
    echo "✅ 数据集已从版本控制中移除"
fi

echo "✅ 修复完成！"
```

保存为 `fix_git.sh`，然后运行：
```bash
chmod +x fix_git.sh
./fix_git.sh
```

---

## 📚 相关文档

- [ENV_SETUP.md](ENV_SETUP.md) - 环境变量配置详解
- [.gitignore](.gitignore) - Git 忽略文件配置
- [GitHub Secret Scanning 文档](https://docs.github.com/en/code-security/secret-scanning)
- [Git LFS 文档](https://git-lfs.github.com/)

---

## ❓ 常见问题

**Q: 我的 Token 已经泄露了怎么办？**
A: 立即到 HuggingFace 撤销该 Token，然后创建新的。

**Q: 数据集文件是否必须提交？**
A: 不需要。数据集太大，建议在 README 中说明如何获取。

**Q: 如何彻底清理 Git 历史？**
A: 使用 `git filter-repo` 或 BFG Repo-Cleaner（参考上面的说明）。

**Q: push 时提示 "Updates were rejected"？**
A: 使用 `git push --force`（⚠️ 谨慎使用，会覆盖远程历史）。

---

**最后更新**: 2026-01-14
