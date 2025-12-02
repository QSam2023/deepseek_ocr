#!/bin/bash
# 快速测试脚本 - 使用最少的训练步数验证完整流程

set -e  # 遇到错误立即退出

echo "========================================"
echo "DeepSeek OCR 快速测试流程"
echo "========================================"
echo ""
echo "这个脚本将："
echo "  1. 检查基础模型（自动下载）"
echo "  2. 划分数据集"
echo "  3. 训练前评估（基线）"
echo "  4. 训练模型（仅 10 步，快速测试）"
echo "  5. 训练后评估（LoRA）"
echo "  6. 对比结果"
echo ""
echo "预计耗时: 5-15 分钟（取决于硬件）"
echo ""

# 询问用户是否继续
read -p "是否继续？[y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消"
    exit 0
fi

# 执行快速测试
echo ""
echo "开始执行..."
echo ""

python train_and_evaluate.py \
    --auto_download_model \
    --data_type stamp \
    --max_steps 10 \
    --summary_file quick_test_summary.json

echo ""
echo "========================================"
echo "✓ 快速测试完成！"
echo "========================================"
echo ""
echo "结果位置："
echo "  - 训练前结果: baseline_result/test/"
echo "  - 训练后结果: lora_result/test/"
echo "  - LoRA 模型: lora_model/"
echo "  - 实验总结: quick_test_summary.json"
echo ""
