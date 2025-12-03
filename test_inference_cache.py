#!/usr/bin/env python3
"""
测试推理缓存功能
对比有缓存 vs 无缓存的推理速度
"""

import os
import sys
import time
import json
import argparse


def test_inference_with_cache():
    """测试推理时使用缓存"""
    print("\n" + "=" * 80)
    print("🧪 推理缓存功能测试")
    print("=" * 80)

    # 查找测试数据
    test_json = "ocr_data/splited_data/table_ocr_test.json"
    if not os.path.exists(test_json):
        print(f"✗ 测试数据不存在: {test_json}")
        print("请先运行: python split_ocr_data.py --data_type table --preprocess")
        return

    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if not test_data:
        print("✗ 测试数据为空")
        return

    # 找到第一个有预处理缓存的样本
    test_sample = None
    for item in test_data:
        if 'preprocessed_path' in item and os.path.exists(item['preprocessed_path']):
            test_sample = item
            break

    if not test_sample:
        print("✗ 没有找到预处理缓存")
        print("请先运行: python split_ocr_data.py --data_type table --preprocess")
        return

    img_path = test_sample['image_path']
    preprocessed_path = test_sample['preprocessed_path']
    task_type = test_sample['task_type']

    print(f"测试图片: {img_path}")
    print(f"缓存路径: {preprocessed_path}")
    print(f"任务类型: {task_type}")

    # 加载模型
    print("\n⏳ 加载模型...")
    from batch_inference import load_local_model, call_local_model

    model_path = "./deepseek_ocr"
    if not os.path.exists(model_path):
        print(f"✗ 模型路径不存在: {model_path}")
        return

    start = time.time()
    model, tokenizer = load_local_model(model_path, None)
    load_time = time.time() - start
    print(f"✓ 模型加载完成，耗时: {load_time:.2f} 秒")

    # 测试 1: 不使用缓存
    print("\n" + "-" * 80)
    print("🔥 测试 1: 实时处理（不使用缓存）")
    print("-" * 80)

    times_no_cache = []
    for i in range(3):
        print(f"\n  运行 {i+1}/3...")
        start = time.time()
        try:
            result_no_cache = call_local_model(
                img_path=img_path,
                task_type=task_type,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2048,
                preprocessed_path=None  # 不使用缓存
            )
            elapsed = time.time() - start
            times_no_cache.append(elapsed)
            print(f"  ✓ 完成，耗时: {elapsed:.2f} 秒")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            return

    avg_no_cache = sum(times_no_cache) / len(times_no_cache)
    print(f"\n📊 平均推理时间（无缓存）: {avg_no_cache:.2f} 秒")

    # 测试 2: 使用缓存
    print("\n" + "-" * 80)
    print("🚀 测试 2: 使用缓存")
    print("-" * 80)

    times_with_cache = []
    for i in range(3):
        print(f"\n  运行 {i+1}/3...")
        start = time.time()
        try:
            result_with_cache = call_local_model(
                img_path=img_path,
                task_type=task_type,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2048,
                preprocessed_path=preprocessed_path  # 使用缓存
            )
            elapsed = time.time() - start
            times_with_cache.append(elapsed)
            print(f"  ✓ 完成，耗时: {elapsed:.2f} 秒")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            return

    avg_with_cache = sum(times_with_cache) / len(times_with_cache)
    print(f"\n📊 平均推理时间（有缓存）: {avg_with_cache:.2f} 秒")

    # 性能对比
    print("\n" + "=" * 80)
    print("📈 性能对比")
    print("=" * 80)
    print(f"实时处理: {avg_no_cache:.2f} 秒")
    print(f"缓存加载: {avg_with_cache:.2f} 秒")
    print(f"速度提升: {avg_no_cache / avg_with_cache:.1f}x")

    speedup = avg_no_cache / avg_with_cache
    if speedup > 1.5:
        print(f"\n✅ 缓存优化显著！速度提升 {speedup:.1f} 倍")
    elif speedup > 1.1:
        print(f"\n✓ 缓存有效，速度提升 {speedup:.1f} 倍")
    else:
        print(f"\n⚠️  缓存效果不明显，速度提升仅 {speedup:.1f} 倍")

    # 结果验证
    print("\n" + "=" * 80)
    print("✅ 结果验证")
    print("=" * 80)

    print(f"无缓存结果长度: {len(str(result_no_cache))}")
    print(f"有缓存结果长度: {len(str(result_with_cache))}")

    # 显示部分结果
    result_str = str(result_with_cache)
    if len(result_str) > 200:
        print(f"\n结果预览:\n{result_str[:200]}...")
    else:
        print(f"\n结果:\n{result_str}")

    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)


def test_batch_inference():
    """测试批量推理"""
    print("\n" + "=" * 80)
    print("🧪 批量推理缓存测试")
    print("=" * 80)

    test_json = "ocr_data/splited_data/table_ocr_test.json"
    if not os.path.exists(test_json):
        print(f"✗ 测试数据不存在: {test_json}")
        return

    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 统计缓存情况
    total = len(test_data)
    with_cache = sum(1 for item in test_data if 'preprocessed_path' in item and os.path.exists(item.get('preprocessed_path', '')))

    print(f"测试集大小: {total}")
    print(f"有缓存: {with_cache} ({with_cache/total*100:.1f}%)")
    print(f"无缓存: {total - with_cache} ({(total-with_cache)/total*100:.1f}%)")

    if with_cache == 0:
        print("\n⚠️  没有预处理缓存")
        print("请运行: python split_ocr_data.py --data_type table --preprocess")
        return

    print(f"\n💡 批量推理时，{with_cache} 张图片将使用缓存加速")
    print(f"   预计节省时间: {with_cache * 25:.0f} 秒 (~{with_cache * 25 / 60:.1f} 分钟)")

    print("\n使用方法:")
    print("  python batch_inference.py \\")
    print("      --data_type table \\")
    print("      --inference_mode local \\")
    print("      --model_path ./deepseek_ocr")


def main():
    parser = argparse.ArgumentParser(description="测试推理缓存功能")
    parser.add_argument(
        '--batch',
        action='store_true',
        help='测试批量推理'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 推理缓存功能测试")
    print("=" * 80)

    # 主测试：单张图片推理对比
    test_inference_with_cache()

    # 可选：批量推理统计
    if args.batch:
        test_batch_inference()

    print("\n" + "=" * 80)
    print("💡 使用说明")
    print("=" * 80)
    print("\n1. 确保数据已预处理:")
    print("   python split_ocr_data.py --data_type all --preprocess")
    print("\n2. 批量推理时自动使用缓存:")
    print("   python batch_inference.py \\")
    print("       --data_type all \\")
    print("       --inference_mode local \\")
    print("       --model_path ./deepseek_ocr")
    print("\n3. 使用完整流程:")
    print("   python train_and_evaluate.py --skip_data_split")
    print()


if __name__ == "__main__":
    main()
