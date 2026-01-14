#!/bin/bash
# 最安全的修复方法：先验证，再操作，保证代码不变

echo "================================================"
echo "🔒 安全修复：验证后清理 Git 历史"
echo "================================================"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 步骤 0: 说明
echo -e "${BLUE}说明：${NC}"
echo "1. 这个脚本只会清理 Git 历史（.git 目录）"
echo "2. 你的所有代码文件、文档、配置完全不会改动"
echo "3. 清理后会用相同的文件重新创建提交"
echo "4. 唯一的区别：新的提交历史中不包含 Token"
echo ""

# 步骤 1: 显示当前文件状态
echo -e "${YELLOW}步骤 1: 显示当前重要文件${NC}"
echo "这些文件在清理前后完全一致："
echo ""
ls -lh operator_search/test_new.py
ls -lh README.md
ls -lh custom_ops/custom_gemm.cu 2>/dev/null || echo "  (算子文件在子目录中)"
echo ""

# 步骤 2: 创建文件快照（用于验证）
echo -e "${YELLOW}步骤 2: 创建文件快照${NC}"
SNAPSHOT_DIR=".snapshot-$(date +%Y%m%d-%H%M%S)"
mkdir -p $SNAPSHOT_DIR

# 复制关键文件到快照目录
cp -r operator_search $SNAPSHOT_DIR/ 2>/dev/null || true
cp -r bert_inference_acceleration $SNAPSHOT_DIR/ 2>/dev/null || true
cp *.md $SNAPSHOT_DIR/ 2>/dev/null || true
cp *.sh $SNAPSHOT_DIR/ 2>/dev/null || true
cp .gitignore $SNAPSHOT_DIR/ 2>/dev/null || true

echo -e "${GREEN}✓ 文件快照已保存到: $SNAPSHOT_DIR${NC}"
echo "如果有任何问题，可以从这里恢复文件"
echo ""

# 步骤 3: 确认当前修复状态
echo -e "${YELLOW}步骤 3: 验证 Token 已从当前文件中移除${NC}"
if grep -q "hf_.*[A-Za-z0-9]" operator_search/test_new.py; then
    echo -e "${RED}✗ 警告：当前文件中仍有疑似 Token！${NC}"
    echo "请先检查 operator_search/test_new.py"
    exit 1
else
    echo -e "${GREEN}✓ 当前文件中没有硬编码的 Token${NC}"
fi
echo ""

# 步骤 4: 列出将要保留的所有文件
echo -e "${YELLOW}步骤 4: 列出所有将要保留的文件${NC}"
echo "这些文件将在清理后完全保留："
find . -type f -not -path "./.git/*" -not -path "./.snapshot-*/*" -not -path "./bert_inference_acceleration/dataset/*" | head -20
echo "... (还有更多文件)"
echo ""

# 步骤 5: 获取远程仓库信息
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REMOTE_URL" ]; then
    echo -e "${RED}✗ 错误：无法获取远程仓库地址${NC}"
    exit 1
fi
echo "远程仓库: $REMOTE_URL"
echo ""

# 步骤 6: 最终确认
echo -e "${YELLOW}准备执行清理操作${NC}"
echo ""
echo -e "${BLUE}将要执行的操作：${NC}"
echo "1. 备份 .git 目录到 .git.backup-*"
echo "2. 删除 .git 目录（Git 历史）"
echo "3. 重新初始化 Git 仓库"
echo "4. 添加所有当前文件（代码完全不变）"
echo "5. 创建新的提交（不包含 Token）"
echo "6. 强制推送到 GitHub"
echo ""
echo -e "${GREEN}保证：你的所有代码文件内容完全不变！${NC}"
echo -e "${GREEN}保证：已创建文件快照在 $SNAPSHOT_DIR${NC}"
echo ""
echo -e "${RED}⚠️  远程仓库的历史将被替换为干净的历史${NC}"
echo ""
read -p "确认要继续吗？[y/N] " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消操作"
    echo "快照已保存，可以安全删除"
    exit 0
fi

# 步骤 7: 备份 .git
echo -e "\n${YELLOW}步骤 7: 备份 .git 目录${NC}"
GIT_BACKUP=".git.backup-$(date +%Y%m%d-%H%M%S)"
cp -r .git $GIT_BACKUP
echo -e "${GREEN}✓ .git 已备份到: $GIT_BACKUP${NC}"

# 步骤 8: 删除 .git（只删除历史，不删除代码）
echo -e "\n${YELLOW}步骤 8: 删除 Git 历史（保留所有代码文件）${NC}"
rm -rf .git
echo -e "${GREEN}✓ Git 历史已删除${NC}"
echo -e "${BLUE}注意：你的代码文件完全没有改动！${NC}"

# 验证代码文件仍然存在
if [ -f "operator_search/test_new.py" ] && [ -f "README.md" ]; then
    echo -e "${GREEN}✓ 验证：代码文件仍然完好${NC}"
else
    echo -e "${RED}✗ 错误：文件丢失！恢复备份...${NC}"
    rm -rf .git
    mv $GIT_BACKUP .git
    exit 1
fi

# 步骤 9: 重新初始化
echo -e "\n${YELLOW}步骤 9: 重新初始化 Git 仓库${NC}"
git init
git remote add origin $REMOTE_URL
echo -e "${GREEN}✓ Git 仓库已重新初始化${NC}"

# 步骤 10: 添加所有文件（除了大文件）
echo -e "\n${YELLOW}步骤 10: 添加文件${NC}"
echo "排除大文件和数据集..."

# 确保 .gitignore 生效
git add .gitignore
git add -A

echo -e "${GREEN}✓ 所有文件已添加${NC}"

# 显示将要提交的文件
echo -e "\n${BLUE}将要提交的文件预览：${NC}"
git status --short | head -20
echo "..."

# 步骤 11: 创建提交
echo -e "\n${YELLOW}步骤 11: 创建全新提交${NC}"
git commit -m "feat: BERT推理加速项目完整实现

## 项目结构

- operator_search/     算子性能调研
- bert_inference_acceleration/  融合算子实现

## 技术栈

- PyTorch 2.1.0
- CUDA 11.8+
- Python 3.10+

## 核心功能

1. 算子性能调研（4种算子）
2. 融合算子实现（2个融合算子）
3. 完整的测试和文档

## 安全性

- ✅ 从环境变量读取 HF_TOKEN
- ✅ .gitignore 排除敏感文件
- ✅ 不包含硬编码的密钥

详细说明请参考 README.md"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 提交创建成功${NC}"
else
    echo -e "${RED}✗ 提交失败，恢复备份...${NC}"
    rm -rf .git
    mv $GIT_BACKUP .git
    exit 1
fi

# 步骤 12: 验证文件没有改动
echo -e "\n${YELLOW}步骤 12: 最终验证${NC}"
echo "验证关键文件内容未改变..."

if diff -q operator_search/test_new.py $SNAPSHOT_DIR/operator_search/test_new.py >/dev/null 2>&1; then
    echo -e "${GREEN}✓ operator_search/test_new.py - 内容完全一致${NC}"
else
    echo -e "${RED}✗ 文件内容有差异（不应该发生）${NC}"
fi

if diff -q README.md $SNAPSHOT_DIR/README.md >/dev/null 2>&1; then
    echo -e "${GREEN}✓ README.md - 内容完全一致${NC}"
else
    echo -e "${RED}✗ 文件内容有差异（不应该发生）${NC}"
fi

# 步骤 13: 推送
echo -e "\n${YELLOW}步骤 13: 推送到 GitHub${NC}"
echo "使用 --force 推送干净的历史..."

# 确保分支名为 main
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "重命名分支为 main..."
    git branch -M main
fi

git push --force origin main

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}================================================${NC}"
    echo -e "${GREEN}✅ 成功！${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "✓ Git 历史已清理（不包含 Token）"
    echo "✓ 所有代码文件完全保留"
    echo "✓ 远程仓库已更新"
    echo ""
    echo "备份位置："
    echo "  - Git 历史备份: $GIT_BACKUP"
    echo "  - 文件快照: $SNAPSHOT_DIR"
    echo ""
    echo -e "${YELLOW}重要：立即撤销旧的 HF Token！${NC}"
    echo "   https://huggingface.co/settings/tokens"
    echo ""
    echo "验证推送："
    echo "   访问你的 GitHub 仓库检查文件"
else
    echo -e "\n${RED}❌ 推送失败${NC}"
    echo "正在恢复备份..."
    rm -rf .git
    mv $GIT_BACKUP .git
    echo "已恢复到原状态"
    exit 1
fi
