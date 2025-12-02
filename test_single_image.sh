#!/bin/bash
# 快速测试单张图片推理

set -e

echo "========================================"
echo "单张图片推理测试"
echo "========================================"
echo ""

# 检查是否提供了图片路径
if [ $# -eq 0 ]; then
    echo "使用方法: ./test_single_image.sh <图片路径> [模型路径]"
    echo ""
    echo "示例:"
    echo "  ./test_single_image.sh ocr_data/stamp_data/stamp_01/stamp_0001.png"
    echo "  ./test_single_image.sh ocr_data/stamp_data/stamp_01/stamp_0001.png ./lora_model"
    echo ""
    echo "可用的测试图片:"
    echo "  - ocr_data/stamp_data/stamp_01/stamp_0001.png"
    echo "  - ocr_data/table_data/table_01/table_0001.png"
    echo ""
    exit 1
fi

IMAGE_PATH=$1
MODEL_PATH=${2:-"./lora_model"}

# 检查图片是否存在
if [ ! -f "$IMAGE_PATH" ]; then
    echo "错误: 图片不存在: $IMAGE_PATH"
    exit 1
fi

echo "图片路径: $IMAGE_PATH"
echo "模型路径: $MODEL_PATH"
echo ""
echo "开始测试..."
echo ""

# 运行调试脚本
python debug_inference.py \
    --model_path "$MODEL_PATH" \
    --image_path "$IMAGE_PATH"

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
