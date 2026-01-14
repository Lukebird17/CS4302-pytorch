#!/bin/bash

echo "================================================"
echo "🔧 Git 推送问题修复脚本"
echo "================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤 1：检查当前状态
echo -e "\n${YELLOW}步骤 1: 检查 Git 状态${NC}"
git status

# 步骤 2：添加修复后的文件
echo -e "\n${YELLOW}步骤 2: 添加修复后的文件${NC}"
git add operator_search/test_new.py
git add .gitignore
git add ENV_SETUP.md
git add GIT_FIX_GUIDE.md
git add fix_and_push.sh

echo -e "${GREEN}✓ 文件已添加到暂存区${NC}"

# 步骤 3：询问是否移除大文件
echo -e "\n${YELLOW}步骤 3: 处理大文件${NC}"
echo "数据集文件很大 (64MB)，建议从版本控制中移除。"
read -p "是否移除数据集文件？[y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在移除数据集..."
    git rm --cached -r bert_inference_acceleration/dataset/ 2>/dev/null || true
    git rm --cached -r operator_search/output/ 2>/dev/null || true
    echo -e "${GREEN}✓ 数据集已从版本控制中移除${NC}"
else
    echo "跳过移除数据集"
fi

# 步骤 4：提交
echo -e "\n${YELLOW}步骤 4: 提交修改${NC}"
git commit -m "fix: 移除硬编码的 HF Token，改用环境变量

- 从环境变量读取 HF_TOKEN，增强安全性
- 添加 .gitignore 排除敏感文件和大文件
- 创建环境变量配置文档
- 修复路径配置，自动检测项目根目录"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 提交成功${NC}"
else
    echo -e "${RED}✗ 提交失败${NC}"
    exit 1
fi

# 步骤 5：推送
echo -e "\n${YELLOW}步骤 5: 推送到 GitHub${NC}"
echo "准备推送到 origin main..."

git push -u origin main

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}================================================${NC}"
    echo -e "${GREEN}✅ 推送成功！${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "后续步骤："
    echo "1. 到 HuggingFace 撤销旧的 Token (如果需要)"
    echo "2. 在 README 中说明数据集获取方式"
    echo "3. 查看 ENV_SETUP.md 了解环境变量配置"
else
    echo -e "\n${RED}================================================${NC}"
    echo -e "${RED}❌ 推送失败${NC}"
    echo -e "${RED}================================================${NC}"
    echo ""
    echo "可能的原因："
    echo "1. 网络问题 - 重试推送"
    echo "2. 权限问题 - 检查 SSH key 或 HTTPS token"
    echo "3. 远程冲突 - 先 pull 再 push"
    echo ""
    echo "详细修复步骤请查看: GIT_FIX_GUIDE.md"
    exit 1
fi
