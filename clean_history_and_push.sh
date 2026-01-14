#!/bin/bash

echo "================================================"
echo "🔧 清理 Git 历史并推送"
echo "================================================"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 步骤 1: 查看当前历史
echo -e "\n${YELLOW}步骤 1: 查看当前提交历史${NC}"
git log --oneline -10

echo -e "\n${YELLOW}问题 commit: f10f0db33107d365b962d1ef4210bb044496e07f${NC}"
echo "这个 commit 包含了硬编码的 HF Token"

# 步骤 2: 创建备份分支
echo -e "\n${YELLOW}步骤 2: 创建备份分支${NC}"
git branch backup-$(date +%Y%m%d-%H%M%S)
echo -e "${GREEN}✓ 备份分支已创建${NC}"

# 步骤 3: 找到安全的 commit（Token 之前的）
echo -e "\n${YELLOW}步骤 3: 检查 commit 历史${NC}"
PROBLEM_COMMIT="f10f0db33107d365b962d1ef4210bb044496e07f"

# 检查该 commit 是否存在
if git cat-file -e $PROBLEM_COMMIT^{commit} 2>/dev/null; then
    echo "找到问题 commit: $PROBLEM_COMMIT"
    
    # 获取前一个 commit
    SAFE_COMMIT=$(git rev-parse ${PROBLEM_COMMIT}^)
    echo "安全的 commit (Token 之前): $SAFE_COMMIT"
    
    echo -e "\n${YELLOW}选项 A: 重置到安全的 commit（推荐）${NC}"
    echo "这会丢弃包含 Token 的所有后续 commit"
    echo ""
    read -p "是否重置到安全的 commit？[y/N] " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 重置到安全的 commit
        git reset --soft $SAFE_COMMIT
        echo -e "${GREEN}✓ 已重置到安全的 commit${NC}"
        
        # 重新添加所有修改
        echo -e "\n${YELLOW}步骤 4: 重新添加修改${NC}"
        git add -A
        
        # 移除大文件
        echo "移除大文件..."
        git rm --cached -r bert_inference_acceleration/dataset/ 2>/dev/null || true
        git rm --cached -r operator_search/output/ 2>/dev/null || true
        
        # 重新提交
        git commit -m "feat: BERT推理加速项目

模块一：算子性能调研
- 使用 PyTorch Profiler 分析 BERT 算子性能
- 支持 softmax, layernorm, addmm, transpose 四种算子
- 测试 IMDB 和 AG News 数据集

模块二：融合算子实现  
- 实现 gemm_bias_add_layernorm 融合算子
- 实现 gemm_bias_gelu_add_layernorm 融合算子
- 优化技术：双缓冲、Warp Shuffle、向量化、Bank Conflict避免
- 性能提升：内存访问减少50-60%，Kernel启动减少60-70%

环境配置：
- PyTorch 2.1.0
- CUDA 11.8+
- 从环境变量读取 HF_TOKEN（安全）

文档：
- README.md - 完整项目文档
- QUICKSTART.md - 5分钟快速开始
- PROJECT_STRUCTURE.md - 项目结构说明
- ENV_SETUP.md - 环境变量配置
- GIT_FIX_GUIDE.md - Git问题修复指南"
        
        echo -e "${GREEN}✓ 重新提交完成${NC}"
    else
        echo "取消重置"
        exit 1
    fi
else
    echo "问题 commit 不在本地历史中，可以直接处理当前状态"
    
    # 确保当前文件都是修复后的版本
    echo -e "\n${YELLOW}步骤 4: 添加修复后的文件${NC}"
    git add operator_search/test_new.py .gitignore ENV_SETUP.md GIT_FIX_GUIDE.md
    
    # 移除大文件
    git rm --cached -r bert_inference_acceleration/dataset/ 2>/dev/null || true
    git rm --cached -r operator_search/output/ 2>/dev/null || true
    
    # 如果有修改，提交
    if ! git diff --cached --quiet; then
        git commit -m "fix: 移除硬编码的 HF Token 和大文件

- 从环境变量读取 HF_TOKEN
- 添加 .gitignore 排除敏感文件
- 移除大文件（dataset > 50MB）"
    fi
fi

# 步骤 5: 强制推送（覆盖远程历史）
echo -e "\n${YELLOW}步骤 5: 推送到 GitHub${NC}"
echo -e "${RED}⚠️  注意：这将使用 --force 覆盖远程历史${NC}"
echo "远程仓库的历史将被替换为本地的干净历史"
echo ""
read -p "确认要强制推送吗？[y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push --force origin main
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}================================================${NC}"
        echo -e "${GREEN}✅ 推送成功！Git 历史已清理干净${NC}"
        echo -e "${GREEN}================================================${NC}"
        echo ""
        echo "✓ Token 已从历史中移除"
        echo "✓ 大文件已排除"
        echo "✓ 远程仓库历史已更新"
        echo ""
        echo "后续步骤："
        echo "1. 到 HuggingFace 撤销旧的 Token"
        echo "   https://huggingface.co/settings/tokens"
        echo "2. 如果其他人克隆了旧仓库，需要重新克隆"
    else
        echo -e "\n${RED}❌ 推送失败${NC}"
        echo "请检查网络连接和权限"
        exit 1
    fi
else
    echo "取消推送"
    exit 1
fi
