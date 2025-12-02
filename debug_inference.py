#!/usr/bin/env python3
"""
调试推理脚本 - 用于测试本地模型推理
单独测试模型加载和推理逻辑，便于排查问题
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image

# 尝试导入必要的库
try:
    from unsloth import FastVisionModel
    import torch
    from transformers import AutoModel
    UNSLOTH_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装: pip install unsloth torch transformers")
    sys.exit(1)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  警告: peft 未安装，无法加载 LoRA adapter")


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}\n")


def check_model_type(model_path: str) -> str:
    """
    检查模型类型

    Returns:
        'lora_adapter', 'complete_model', or 'unknown'
    """
    if not os.path.exists(model_path):
        return 'unknown'

    adapter_config = os.path.join(model_path, "adapter_config.json")
    config = os.path.join(model_path, "config.json")

    if os.path.exists(adapter_config) and not os.path.exists(config):
        return 'lora_adapter'
    elif os.path.exists(config):
        return 'complete_model'
    else:
        return 'unknown'


def load_model(model_path: str, base_model_path: str = None):
    """加载模型并返回 model 和 tokenizer"""
    print_section("步骤 1: 检查模型类型")

    model_type = check_model_type(model_path)
    print(f"模型路径: {model_path}")
    print(f"模型类型: {model_type}")

    if model_type == 'unknown':
        raise ValueError(f"无法识别模型类型: {model_path}")

    # 设置环境变量
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

    if model_type == 'lora_adapter':
        print("\n检测到 LoRA adapter")

        # 确定基础模型路径
        if base_model_path is None:
            base_model_path = "./deepseek_ocr"
            print(f"使用默认基础模型路径: {base_model_path}")
        else:
            print(f"使用指定基础模型路径: {base_model_path}")

        if not os.path.exists(base_model_path):
            raise ValueError(f"基础模型不存在: {base_model_path}")

        print_section("步骤 2: 加载基础模型")
        print(f"正在加载基础模型: {base_model_path}")
        print("这可能需要一些时间...\n")

        try:
            base_model, tokenizer = FastVisionModel.from_pretrained(
                base_model_path,
                load_in_4bit=False,
                auto_model=AutoModel,
                trust_remote_code=True,
                unsloth_force_compile=True,
                use_gradient_checkpointing="unsloth",
            )
            print("✓ 基础模型加载成功")
            print(f"  模型类型: {type(base_model)}")
            print(f"  Tokenizer 类型: {type(tokenizer)}")

            # 检查是否有 infer 方法
            if hasattr(base_model, 'infer'):
                print("  ✓ 模型有 infer() 方法")
            else:
                print("  ✗ 警告: 模型没有 infer() 方法")

        except Exception as e:
            print(f"✗ 基础模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        if not PEFT_AVAILABLE:
            print("\n✗ 错误: peft 未安装，无法加载 LoRA adapter")
            print("请安装: pip install peft")
            sys.exit(1)

        print_section("步骤 3: 加载 LoRA adapter")
        print(f"正在加载 LoRA adapter: {model_path}\n")

        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✓ LoRA adapter 加载成功")
            print(f"  模型类型: {type(model)}")

            # 检查是否有 infer 方法
            if hasattr(model, 'infer'):
                print("  ✓ 模型仍然有 infer() 方法")
            else:
                print("  ✗ 警告: 模型失去了 infer() 方法")

            # 检查基础模型
            if hasattr(model, 'base_model'):
                print(f"  基础模型类型: {type(model.base_model)}")
                if hasattr(model.base_model, 'infer'):
                    print("  ✓ 基础模型有 infer() 方法")

        except Exception as e:
            print(f"✗ LoRA adapter 加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:  # complete_model
        print("\n检测到完整模型")

        print_section("步骤 2: 加载完整模型")
        print(f"正在加载模型: {model_path}")
        print("这可能需要一些时间...\n")

        try:
            model, tokenizer = FastVisionModel.from_pretrained(
                model_path,
                load_in_4bit=False,
                auto_model=AutoModel,
                trust_remote_code=True,
                unsloth_force_compile=True,
                use_gradient_checkpointing="unsloth",
            )
            print("✓ 模型加载成功")
            print(f"  模型类型: {type(model)}")
            print(f"  Tokenizer 类型: {type(tokenizer)}")

            # 检查是否有 infer 方法
            if hasattr(model, 'infer'):
                print("  ✓ 模型有 infer() 方法")
            else:
                print("  ✗ 警告: 模型没有 infer() 方法")

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # 启用推理模式
    print_section("启用推理模式")
    print("设置模型为推理模式...")
    try:
        FastVisionModel.for_inference(model)
        print("✓ 推理模式已启用")
        print("  - 模型已设置为 eval() 模式")
        print("  - 梯度计算已禁用")
        print("  - 推理优化已应用")
    except Exception as e:
        print(f"⚠️  警告: 无法启用推理模式: {e}")
        print("  将继续使用默认模式")

    print_section("模型加载完成")
    return model, tokenizer


def test_inference(model, tokenizer, image_path: str, prompt: str = None):
    """测试模型推理"""
    print_section("步骤 4: 测试推理")

    # 检查图片是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    print(f"图片路径: {image_path}")

    # 检查图片是否可读
    try:
        img = Image.open(image_path)
        print(f"图片尺寸: {img.size}")
        print(f"图片模式: {img.mode}")
    except Exception as e:
        raise ValueError(f"无法读取图片: {e}")

    # 默认 prompt
    if prompt is None:
        prompt = "<image>\nFree OCR. Please extract all text from the image."

    print(f"\nPrompt: {prompt}")

    # 详细检查模型结构
    print(f"\n详细检查模型结构:")
    print(f"  模型类型: {type(model)}")
    print(f"  hasattr(model, 'infer'): {hasattr(model, 'infer')}")

    if hasattr(model, 'base_model'):
        print(f"  有 base_model 属性")
        print(f"  base_model 类型: {type(model.base_model)}")
        print(f"  hasattr(model.base_model, 'infer'): {hasattr(model.base_model, 'infer')}")

        if hasattr(model.base_model, 'model'):
            print(f"  有 base_model.model 属性")
            print(f"  base_model.model 类型: {type(model.base_model.model)}")
            print(f"  hasattr(model.base_model.model, 'infer'): {hasattr(model.base_model.model, 'infer')}")

    # 列出模型的所有方法
    print(f"\n模型的主要方法:")
    for attr in dir(model):
        if not attr.startswith('_') and callable(getattr(model, attr, None)):
            if 'infer' in attr.lower() or 'generate' in attr.lower() or 'forward' in attr.lower():
                print(f"  - {attr}")

    # 尝试推理
    print("\n开始推理...")
    print("-" * 80)

    result = None

    # 创建临时输出目录（即使不保存文件也需要）
    import tempfile
    import shutil
    temp_output_dir = tempfile.mkdtemp(prefix='deepseek_ocr_debug_')
    print(f"临时输出目录: {temp_output_dir}")

    try:
        # 调用 infer() 方法，使用 eval_mode=True 直接获取返回值
        print("\n调用 model.infer()...")
        print(f"  图片: {image_path}")
        print(f"  参数: eval_mode=True, save_results=False")

        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=temp_output_dir,  # 必须提供
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=False,
            eval_mode=True  # 关键：直接返回结果
        )
        print("✓ infer() 调用完成")

        # 检查返回值
        print("\n检查返回值...")
        print(f"  返回值类型: {type(result)}")
        print(f"  是否为 None: {result is None}")

        if result is not None:
            if isinstance(result, str):
                print(f"  返回值长度: {len(result)} 字符")
            elif isinstance(result, list):
                print(f"  返回值是列表，长度: {len(result)}")
                if result:
                    result = result[0]  # 取第一个元素
                    print(f"  取第一个元素: {len(result)} 字符")

        print("-" * 80)

        if result and str(result).strip():
            print(f"✓ 推理完成，返回 {len(str(result))} 字符")
        else:
            print("✗ 返回了空结果或 None")

    except Exception as e:
        print("-" * 80)
        print(f"✗ 推理过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 清理临时目录
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            print(f"✓ 已清理临时目录")

    # 检查结果
    print_section("步骤 5: 检查结果")

    print(f"结果类型: {type(result)}")
    print(f"结果是否为 None: {result is None}")

    if result is None:
        print("\n✗ 错误: 推理返回 None")
        print("\n可能的原因:")
        print("  1. 模型推理内部失败")
        print("  2. 图片格式不支持")
        print("  3. 模型配置问题")
        print("  4. LoRA adapter 与基础模型不兼容")
        return None

    if isinstance(result, str):
        print(f"结果长度: {len(result)} 字符")
        print(f"\n原始结果:")
        print("-" * 80)
        print(result)
        print("-" * 80)

        # 尝试解析 JSON
        print("\n尝试解析 JSON:")
        json_text = result.strip()

        # 去除 markdown 标记
        if json_text.startswith("```json"):
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif json_text.startswith("```"):
            json_text = json_text.split("```")[1].split("```")[0].strip()

        try:
            result_json = json.loads(json_text)
            print("✓ JSON 解析成功")
            print("\n解析后的 JSON:")
            print(json.dumps(result_json, ensure_ascii=False, indent=2))
        except json.JSONDecodeError as e:
            print(f"✗ JSON 解析失败: {e}")
            print(f"\n尝试解析的文本:")
            print("-" * 80)
            print(json_text)
            print("-" * 80)
    else:
        print(f"结果: {result}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="调试本地模型推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 测试完整模型
  python debug_inference.py \\
      --model_path ./deepseek_ocr \\
      --image_path ocr_data/stamp_data/stamp_01/stamp_0001.png

  # 测试 LoRA 模型
  python debug_inference.py \\
      --model_path ./lora_model \\
      --image_path ocr_data/stamp_data/stamp_01/stamp_0001.png

  # 指定基础模型和自定义 prompt
  python debug_inference.py \\
      --model_path ./lora_model \\
      --base_model_path ./deepseek_ocr \\
      --image_path test.jpg \\
      --prompt "<image>\\nExtract all text."
        """
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（可以是完整模型或 LoRA adapter）')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='基础模型路径（当 model_path 是 LoRA adapter 时需要）')
    parser.add_argument('--image_path', type=str, required=True,
                        help='测试图片路径')
    parser.add_argument('--prompt', type=str, default=None,
                        help='自定义 prompt（默认使用 Free OCR）')

    args = parser.parse_args()

    print("=" * 80)
    print("DeepSeek OCR 推理调试脚本")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  模型路径: {args.model_path}")
    if args.base_model_path:
        print(f"  基础模型路径: {args.base_model_path}")
    print(f"  图片路径: {args.image_path}")
    if args.prompt:
        print(f"  自定义 prompt: {args.prompt}")

    try:
        # 加载模型
        model, tokenizer = load_model(args.model_path, args.base_model_path)

        # 测试推理
        result = test_inference(model, tokenizer, args.image_path, args.prompt)

        # 总结
        print_section("总结")
        if result is not None:
            print("✓ 推理成功")
            print("\n如果看到正确的结果，说明模型和推理逻辑都正常。")
            print("如果结果不正确，可能是 prompt 或模型训练的问题。")
        else:
            print("✗ 推理失败")
            print("\n请检查:")
            print("  1. 模型是否正确加载")
            print("  2. 模型是否有 infer() 方法")
            print("  3. 图片格式是否支持")
            print("  4. LoRA adapter 是否与基础模型兼容")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
