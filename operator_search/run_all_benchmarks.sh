#!/bin/bash

# 定义基础路径和算子列表
BASE_DIR="/hy-tmp/fj"
OPERATORS=("softmax" "layernorm" "addmm" "transpose")
LOG_FILE="${BASE_DIR}/benchmark_exec.log"

# 清理旧日志
echo "=== BERT Operator Benchmark Session Start: $(date) ===" | tee $LOG_FILE

# 循环执行每个算子的测试
for OP in "${OPERATORS[@]}"
do
    echo -e "\n\033[1;32m[STEP] 开始测试算子: ${OP} ...\033[0m" | tee -a $LOG_FILE
    echo "----------------------------------------------------" | tee -a $LOG_FILE
    
    # 执行 Python 脚本
    # 使用 python -u 参数确保日志实时刷新到控制台
    python -u test_new.py --op "${OP}" 2>&1 | tee -a $LOG_FILE
    
    if [ $? -eq 0 ]; then
        echo -e "\033[1;34m[SUCCESS] ${OP} 测试完成，数据已保存至 ./${OP}/ 文件夹。\033[0m" | tee -a $LOG_FILE
    else
        echo -e "\033[1;31m[ERROR] ${OP} 测试过程中出现异常，请检查日志。\033[0m" | tee -a $LOG_FILE
    fi
    
    echo "----------------------------------------------------" | tee -a $LOG_FILE
done

echo -e "\n\033[1;35m[FINISH] 所有算子测试任务已完成！\033[0m" | tee -a $LOG_FILE
echo "完整执行日志请查看: ${LOG_FILE}"
