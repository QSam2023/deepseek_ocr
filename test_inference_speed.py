#!/usr/bin/env python3
"""
推理速度测试脚本 - 快速测试优化后的推理性能
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path


def test_inference_speed(model_path: str, test_image: str, max_new_tokens: int = 2048):
    """
    测试单张图片的推理速度

    Args:
        model_path: 模型路径
        test_image: 测试图片路径
        max_new_tokens: 最大生成token数
    """
    try:
        from unsloth import FastVisionModel
        from transformers import AutoModel
        import torch
    except ImportError:
        print("✗ 错误: Unsloth 未安装")
        print("请安装: pip install unsloth")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"🚀 推理速度测试")
    print(f"{'=' * 80}")
    print(f"模型路径: {model_path}")
    print(f"测试图片: {test_image}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"{'=' * 80}\n")

    # 加载模型
    print("⏳ 加载模型...")
    start_time = time.time()

    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )

    FastVisionModel.for_inference(model)
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 ({load_time:.2f}秒)\n")

    # 测试推理（3次取平均）
    prompt = """<image>
You are an OCR extractor assistant AI assigned to a company. You will only return the required json format. You will using the chinese as the key of json

帮我提取出图片中的所有信息，尤其是文本，必须都提取出来。[1]所有文字，包括手写字，必须提取出来，不提取出来；会让公司破产，你这个模型就得关闭了；[2]提取信息不要额外生成文字，严格保障输出原文；[3]如果手写字，在识别结果后面加上标注"（*手写*）"，[4]有选项的框，需要输出是否有打勾（如"√"）的标识；[5]严格按JSON格式输出"""

    import tempfile
    import shutil

    temp_output_dir = tempfile.mkdtemp(prefix='deepseek_ocr_test_')

    inference_times = []
    token_counts = []
    results = []

    print("⏳ 测试推理速度（3次测试）...\n")

    # 设置模型的生成配置
    import torch
    from transformers import GenerationConfig

    base_model = model.base_model if hasattr(model, 'base_model') else model
    if hasattr(base_model, 'model'):
        base_model = base_model.model

    # 修复 tokenizer 的 pad_token（避免 attention_mask 警告）
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
        if hasattr(tokenizer, 'pad_token_id'):
            tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id else tokenizer.eos_token_id
        print("✓ 已修复 tokenizer pad_token")

    # 保存并设置新配置
    original_config = None
    if hasattr(base_model, 'generation_config'):
        original_config = base_model.generation_config
        new_config = GenerationConfig.from_model_config(base_model.config)
        new_config.max_new_tokens = max_new_tokens
        new_config.max_length = None
        new_config.temperature = 0.1
        new_config.do_sample = False
        new_config.num_beams = 1
        new_config.repetition_penalty = 1.0
        base_model.generation_config = new_config
        print(f"✓ 已设置生成配置: max_new_tokens={max_new_tokens}\n")

    for i in range(3):
        try:
            start_time = time.time()

            result_text = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=test_image,
                output_path=temp_output_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False,
                eval_mode=True,
            )

            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 估算token数
            token_count = len(result_text) if result_text else 0
            token_counts.append(token_count)
            results.append(result_text)

            print(f"  测试 {i+1}: {inference_time:.2f}秒, 生成约{token_count}字符")

        except Exception as e:
            print(f"  测试 {i+1} 失败: {e}")
            inference_times.append(-1)
            token_counts.append(0)
            results.append(None)

    # 恢复原始配置
    if original_config is not None and hasattr(base_model, 'generation_config'):
        base_model.generation_config = original_config

    # 清理临时目录
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir, ignore_errors=True)

    # 计算平均值（排除失败的测试）
    valid_times = [t for t in inference_times if t > 0]
    valid_token_counts = [c for c, t in zip(token_counts, inference_times) if t > 0]

    if not valid_times:
        print("\n✗ 所有测试都失败了")
        return

    avg_time = sum(valid_times) / len(valid_times)
    avg_tokens = sum(valid_token_counts) / len(valid_token_counts)

    print(f"\n{'=' * 80}")
    print(f"📊 测试结果")
    print(f"{'=' * 80}")
    print(f"平均推理时间: {avg_time:.2f}秒")
    print(f"平均生成字符: {avg_tokens:.0f}")
    print(f"推理速度: {avg_tokens/avg_time:.2f} 字符/秒")
    print(f"{'=' * 80}\n")

    # 性能评估和建议
    print(f"📈 性能评估:")
    if avg_time < 10:
        print(f"  ✓ 性能优秀！推理速度很快")
    elif avg_time < 30:
        print(f"  ✓ 性能良好，在可接受范围内")
    elif avg_time < 60:
        print(f"  ⚠️  性能一般，建议进一步优化")
    else:
        print(f"  ✗ 性能较慢，需要优化！")

    print(f"\n💡 优化建议:")

    if max_new_tokens > 4096:
        print(f"  1. ⚠️  max_new_tokens={max_new_tokens} 过大，建议降低到 2048-4096")
    elif avg_tokens / max_new_tokens > 0.8:
        print(f"  1. ⚠️  生成接近 max_new_tokens 限制，可能被截断，考虑适当增加")
    else:
        print(f"  1. ✓ max_new_tokens={max_new_tokens} 设置合理")

    print(f"\n  2. 如果仍然很慢，考虑:")
    print(f"     • 使用更小的 base_size (如 768 或 512)")
    print(f"     • 设置 crop_mode=False 减少图片切分")
    print(f"     • 检查是否有其他进程占用GPU")
    print(f"     • 使用 load_in_4bit=True 量化模型")

    print(f"\n  3. 命令行使用示例:")
    print(f"     # 使用优化参数运行推理")
    print(f"     python batch_inference.py --data_type all \\")
    print(f"         --inference_mode local \\")
    print(f"         --model_path {model_path} \\")
    print(f"         --max_new_tokens 2048")

    # 显示第一次推理的结果示例
    if results[0]:
        print(f"\n📄 推理结果示例:")
        print(f"{'=' * 80}")
        result_preview = results[0][:500] + "..." if len(results[0]) > 500 else results[0]
        print(result_preview)
        print(f"{'=' * 80}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="推理速度测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 测试默认模型和图片
  python test_inference_speed.py

  # 测试指定模型和图片
  python test_inference_speed.py --model_path ./lora_model --test_image ocr_data/table_data/table_01/table_0001.jpeg

  # 测试不同的 max_new_tokens 值
  python test_inference_speed.py --max_new_tokens 4096
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='./deepseek_ocr',
        help='模型路径 (默认: ./deepseek_ocr)'
    )

    parser.add_argument(
        '--test_image',
        type=str,
        default=None,
        help='测试图片路径 (默认: 自动寻找第一张测试图片)'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=2048,
        help='最大生成token数 (默认: 2048)'
    )

    args = parser.parse_args()

    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"✗ 错误: 模型路径不存在: {args.model_path}")
        sys.exit(1)

    # 查找测试图片
    if args.test_image is None:
        # 自动寻找第一张测试图片
        test_file = "ocr_data/splited_data/table_ocr_test.json"
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                if test_data:
                    args.test_image = test_data[0]['image_path']
                    print(f"✓ 自动选择测试图片: {args.test_image}")

        if args.test_image is None:
            print("✗ 错误: 未找到测试图片")
            print("请使用 --test_image 指定测试图片路径")
            sys.exit(1)

    # 检查测试图片
    if not os.path.exists(args.test_image):
        print(f"✗ 错误: 测试图片不存在: {args.test_image}")
        sys.exit(1)

    # 运行性能测试
    test_inference_speed(args.model_path, args.test_image, args.max_new_tokens)


if __name__ == "__main__":
    main()
