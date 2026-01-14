#!/bin/bash
echo "========================================"
echo "修复Git历史中的HF Token"
echo "========================================"

# 1. 备份当前分支
git branch backup-before-fix

echo "✓ 已创建备份分支: backup-before-fix"
echo ""
echo "方案: 使用交互式rebase修改commit 3f97537"
echo ""
echo "步骤:"
echo "1. git rebase -i 67daf86e5e"
echo "2. 将 3f97537 那行的 'pick' 改为 'edit'"
echo "3. 保存退出"
echo "4. 修改 public/test.py 删除token"
echo "5. git add public/test.py"
echo "6. git commit --amend --no-edit"
echo "7. git rebase --continue"
echo "8. git push --force-with-lease origin main"
echo ""
echo "或者使用自动脚本 (更安全):"
echo "========================================"

