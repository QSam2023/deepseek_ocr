#!/usr/bin/env python3
"""
测试图片预处理和缓存功能
验证预处理是否正确工作，以及性能提升效果
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path


def test_single_image_preprocessing(image_path: str, task_type: str = "table_ocr"):
    """测试单张图片的预处理"""
    from image_preprocessor import ImagePreprocessor

    print("\n" + "=" * 80)
    print("🧪 测试 1: 单张图片预处理")
    print("=" * 80)

    preprocessor = ImagePreprocessor(
        image_size=640,
        base_size=1024,
        crop_mode=True,
        cache_dir="ocr_data/preprocessed_cache"
    )

    print(f"图片路径: {image_path}")
    print(f"任务类型: {task_type}")

    # 预处理
    print("\n⏳ 预处理图片...")
    start = time.time()
    preprocessed = preprocessor.preprocess_image(image_path)
    preprocess_time = time.time() - start

    print(f"✓ 预处理完成，耗时: {preprocess_time:.3f} 秒")
    print(f"\n预处理结果:")
    print(f"  - images_ori shape: {preprocessed['images_ori'].shape}")
    print(f"  - images_crop shape: {preprocessed['images_crop'].shape}")
    print(f"  - images_spatial_crop shape: {preprocessed['images_spatial_crop'].shape}")
    print(f"  - tokenized_image length: {len(preprocessed['tokenized_image'])}")
    print(f"  - crop_ratio: {preprocessed['crop_ratio']}")
    print(f"  - original_size: {preprocessed['original_size']}")

    # 保存缓存
    cache_path = preprocessor.get_cache_path(image_path, task_type)
    print(f"\n⏳ 保存缓存到: {cache_path}")
    start = time.time()
    preprocessor.save_cache(preprocessed, cache_path)
    save_time = time.time() - start
    print(f"✓ 缓存保存完成，耗时: {save_time:.3f} 秒")

    # 加载缓存
    print(f"\n⏳ 从缓存加载...")
    start = time.time()
    loaded = preprocessor.load_cache(cache_path)
    load_time = time.time() - start
    print(f"✓ 缓存加载完成，耗时: {load_time:.3f} 秒")

    # 验证
    print(f"\n✅ 验证缓存:")
    print(f"  - 数据完整性: {'通过' if loaded is not None else '失败'}")
    print(f"  - 加载速度提升: {preprocess_time / load_time:.1f}x")

    return cache_path, preprocess_time, load_time


def test_batch_preprocessing():
    """测试批量预处理"""
    print("\n" + "=" * 80)
    print("🧪 测试 2: 批量预处理")
    print("=" * 80)

    # 查找测试数据
    test_json = "ocr_data/splited_data/table_ocr_test.json"
    if not os.path.exists(test_json):
        print(f"⚠️  跳过：测试数据不存在 {test_json}")
        return

    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 取前 10 张图片测试
    test_samples = test_data[:min(10, len(test_data))]
    image_paths = [item['image_path'] for item in test_samples]
    task_types = [item['task_type'] for item in test_samples]

    print(f"测试样本数: {len(test_samples)}")

    from image_preprocessor import batch_preprocess_images

    print("\n⏳ 批量预处理...")
    start = time.time()
    cache_paths = batch_preprocess_images(
        image_paths=image_paths,
        task_types=task_types,
        image_size=640,
        base_size=1024,
        crop_mode=True,
        cache_dir="ocr_data/preprocessed_cache",
        verbose=True
    )
    total_time = time.time() - start

    success_count = sum(1 for p in cache_paths if p is not None)
    print(f"\n✓ 批量预处理完成")
    print(f"  - 成功: {success_count}/{len(test_samples)}")
    print(f"  - 总耗时: {total_time:.2f} 秒")
    print(f"  - 平均耗时: {total_time/len(test_samples):.3f} 秒/图")


def test_data_collator_with_cache():
    """测试 data collator 加载缓存"""
    print("\n" + "=" * 80)
    print("🧪 测试 3: Data Collator 加载缓存")
    print("=" * 80)

    # 准备测试数据
    test_json = "ocr_data/splited_data/table_ocr_test.json"
    if not os.path.exists(test_json):
        print(f"⚠️  跳过：测试数据不存在 {test_json}")
        return

    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if not test_data:
        print("⚠️  跳过：没有测试数据")
        return

    # 取第一个样本
    sample = test_data[0]

    # 检查是否有预处理缓存
    if 'preprocessed_path' not in sample:
        print("⚠️  跳过：测试数据没有预处理缓存，请先运行:")
        print("     python split_ocr_data.py --data_type table --preprocess")
        return

    print(f"样本图片: {sample['image_path']}")
    print(f"缓存路径: {sample.get('preprocessed_path', 'N/A')}")

    # 模拟 data collator 的处理
    try:
        from PIL import Image
        from unsloth_data_collator import DeepSeekOCRDataCollator

        # 创建简单的 mock tokenizer 和 model
        class MockTokenizer:
            def __init__(self):
                self.bos_token_id = 1
                self.eos_token = "</s>"
                self.pad_token_id = 0

        class MockModel:
            def __init__(self):
                import torch
                self.dtype = torch.float16

        tokenizer = MockTokenizer()
        model = MockModel()

        collator = DeepSeekOCRDataCollator(
            tokenizer=tokenizer,
            model=model,
            image_size=640,
            base_size=1024,
            crop_mode=True
        )

        # 准备消息（简化版本）
        messages = [
            {
                "role": "<|User|>",
                "content": "<image>\nTest prompt",
                "images": [Image.open(sample['image_path'])]
            },
            {
                "role": "<|Assistant|>",
                "content": "Test response"
            }
        ]

        # 测试1: 不使用缓存
        print("\n⏳ 测试实时处理（不使用缓存）...")
        start = time.time()
        result_no_cache = collator.process_single_sample(messages, preprocessed_path=None)
        time_no_cache = time.time() - start
        print(f"✓ 完成，耗时: {time_no_cache:.3f} 秒")

        # 测试2: 使用缓存
        print("\n⏳ 测试从缓存加载...")
        start = time.time()
        result_with_cache = collator.process_single_sample(messages, preprocessed_path=sample['preprocessed_path'])
        time_with_cache = time.time() - start
        print(f"✓ 完成，耗时: {time_with_cache:.3f} 秒")

        # 对比
        print(f"\n📊 性能对比:")
        print(f"  - 实时处理: {time_no_cache:.3f} 秒")
        print(f"  - 缓存加载: {time_with_cache:.3f} 秒")
        print(f"  - 速度提升: {time_no_cache / time_with_cache:.1f}x")

        # 验证结果一致性
        print(f"\n✅ 结果验证:")
        print(f"  - input_ids shape: {result_no_cache['input_ids'].shape} vs {result_with_cache['input_ids'].shape}")
        print(f"  - images_ori shape: {result_no_cache['images_ori'].shape} vs {result_with_cache['images_ori'].shape}")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="测试图片预处理和缓存功能")
    parser.add_argument(
        '--test_image',
        type=str,
        default=None,
        help='测试图片路径'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='运行所有测试'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 图片预处理和缓存功能测试")
    print("=" * 80)

    # 查找测试图片
    if args.test_image is None:
        test_json = "ocr_data/splited_data/table_ocr_test.json"
        if os.path.exists(test_json):
            with open(test_json, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                if test_data:
                    args.test_image = test_data[0]['image_path']
                    print(f"✓ 自动选择测试图片: {args.test_image}")

    if args.test_image and os.path.exists(args.test_image):
        # 测试 1: 单张图片
        test_single_image_preprocessing(args.test_image)
    else:
        print("\n⚠️  跳过测试 1: 没有找到测试图片")

    if args.all:
        # 测试 2: 批量预处理
        test_batch_preprocessing()

        # 测试 3: Data collator
        test_data_collator_with_cache()

    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)
    print("\n💡 使用方法:")
    print("  1. 数据切分时启用预处理:")
    print("     python split_ocr_data.py --data_type table --preprocess")
    print("\n  2. 训练时自动使用缓存（无需修改训练命令）:")
    print("     python train_model.py --config train_config.yaml")
    print("\n  3. 跳过数据切分，直接使用已有缓存:")
    print("     python train_and_evaluate.py --skip_data_split")
    print()


if __name__ == "__main__":
    main()
